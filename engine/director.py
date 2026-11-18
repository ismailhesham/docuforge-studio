import json
import re
from openai import OpenAI

SYSTEM_PROMPT = """You are an elite documentary film director & editor.
Convert the provided English narration segment into rapid-fire documentary shots.
CRITICAL RULES:
1. NEVER alter, summarize, or drop words from user's narration when mode='script'. Split verbatim text into 10-16 word shots (~2.5s-3.8s each).
2. 80-85% of shots MUST have media_type="video".
3. 15-20% of shots may use photos ONLY in back-to-back PAIRS: first media_type="photo_in", immediately followed by media_type="photo_out". NEVER output a single isolated photo or 3 photos in a row.
4. Graphic overlays ('overlay' field) MUST be sparse (max 2-3 per 9 shots; first 2 shots MUST be 'clean'):
   - "clean": full-bleed footage, no text.
   - "mustard_split": major dramatic turn/revelation. Provide 'headline' (1-2 impact words, CAPS) & 'subtext' (2-4 words).
   - "lower_pill": first mention of technical/historical entity. Provide 'pill_title' & 'pill_sub'. Never repeat within 4 shots.
   - "dual_compare": contrast old vs new / A vs B. Provide 'compare_title', 'left_label', 'right_label', plus 'secondary_yt_query'.
   - "rolling_stat": concrete big number/metric. Provide 'stat_target' (number), 'stat_suffix' (e.g. '%', '+ HP', 'B'), 'stat_label'.
5. Every shot needs:
   - "spoken": exact narration sentence/clause
   - "media_type": "video" | "photo_in" | "photo_out"
   - "overlay": "clean" | "mustard_split" | "lower_pill" | "dual_compare" | "rolling_stat"
   - "visual_prompt": concrete literal visual description for CLIP vision check (no abstract metaphors)
   - "yt_queries": list of 3 specific YouTube search queries (e.g. "robotic car factory welding 4k b roll")
   - "wiki_queries": list of 2 Wikimedia Commons photo search terms
   - "start_sec": integer >= 15 (skip channel intros)
Return ONLY valid JSON: {"shots": [ ... ]}"""


def _chunk_script_by_words(text: str, max_words: int = 340):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur, count = [], [], 0
    for s in sentences:
        if not s.strip():
            continue
        w = len(s.split())
        if cur and count + w > max_words:
            chunks.append(" ".join(cur))
            cur, count = [s], w
        else:
            cur.append(s)
            count += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def enforce_editorial_rhythm(shots: list) -> list:
    """حارس برمجي يضمن قاعدة أزواج الصور المتعاكسة وعدم تكرار الكروت المنسدلة."""
    n = len(shots)
    for idx, sh in enumerate(shots):
        sh["id"] = idx + 1
        sh.setdefault("media_type", "video")
        sh.setdefault("overlay", "clean")
        sh.setdefault("start_sec", 16)
        if int(sh.get("start_sec", 16)) < 14:
            sh["start_sec"] = 16
        sh.setdefault("yt_queries", [sh.get("visual_prompt", "cinematic documentary footage")])
        sh.setdefault("wiki_queries", [sh.get("visual_prompt", "industrial engineering")])

    # إصلاح أزواج الصور: كل photo_in لازم يتبعها photo_out فوراً وإلا تتحول لفيديو
    i = 0
    while i < n:
        mt = shots[i]["media_type"]
        if mt in ("photo_in", "photo_out"):
            if i + 1 < n and shots[i + 1]["media_type"] in ("photo_in", "photo_out"):
                shots[i]["media_type"] = "photo_in"
                shots[i + 1]["media_type"] = "photo_out"
                # لو فيه صورة تالتة وراهم نحولها فيديو
                if i + 2 < n and shots[i + 2]["media_type"] in ("photo_in", "photo_out"):
                    shots[i + 2]["media_type"] = "video"
                i += 2
                continue
            else:
                shots[i]["media_type"] = "video"
        i += 1

    # منع تكرار lower_pill في أقل من 4 لقطات وتنظيف أول لقطتين
    last_pill = -10
    for idx, sh in enumerate(shots):
        if idx < 2 and sh["overlay"] != "clean":
            sh["overlay"] = "clean"
        if sh["overlay"] == "lower_pill":
            if idx - last_pill < 4:
                sh["overlay"] = "clean"
            else:
                last_pill = idx
    return shots


def build_storyboard(api_key: str, text_input: str, mode: str = "script", target_minutes: int = 5, progress_cb=None) -> list:
    client = OpenAI(api_key=api_key.strip(), base_url="https://api.deepseek.com")

    if mode == "topic":
        if progress_cb:
            progress_cb(0.03, "جاري كتابة الاسكربت الوثائقي الكامل عبر DeepSeek...")
        target_words = max(250, int(target_minutes * 145))
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Write gripping, investigative high-tech/documentary YouTube narration in English. Pure spoken prose only, no headings or stage notes."},
                {"role": "user", "content": f"Topic: {text_input}\nTarget word count: ~{target_words} words."}
            ],
            temperature=0.7,
        )
        full_script = resp.choices[0].message.content.strip()
    else:
        full_script = text_input.strip()

    chunks = _chunk_script_by_words(full_script, max_words=330)
    all_shots = []
    for c_idx, chunk in enumerate(chunks):
        if progress_cb:
            frac = 0.04 + 0.08 * ((c_idx + 1) / max(1, len(chunks)))
            progress_cb(frac, f"إخراج وتقسيم الدفعة السردية {c_idx + 1} من {len(chunks)} عبر DeepSeek...")

        resp = client.chat.completions.create(
            model="deepseek-chat",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Mode: script (keep exact words). Batch {c_idx+1}/{len(chunks)}:\n\n{chunk}"}
            ],
            temperature=0.35,
        )
        data = json.loads(resp.choices[0].message.content)
        batch_shots = data.get("shots", [])
        all_shots.extend(batch_shots)

    return enforce_editorial_rhythm(all_shots)
