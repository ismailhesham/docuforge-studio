import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
import torch
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPProcessor
from config import FPS, HEADERS, HEIGHT, LOGO_SAFE_MARGIN, WIDTH

NEGATIVE_PROMPTS = [
    "youtuber selfie talking head vlog",
    "breaking news lower third tv banner ticker",
    "large stock footage watermark logo across center",
    "video game screen hud interface",
]


class ClipGatekeeper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lock = threading.Lock()

    def score_image(self, pil_img: Image.Image, target_prompt: str) -> float:
        texts = [target_prompt] + NEGATIVE_PROMPTS
        with self.lock, torch.no_grad():
            inputs = self.processor(text=texts, images=pil_img, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inputs).logits_per_image[0]
            probs = logits.softmax(dim=0).cpu().numpy()
        pos = float(probs[0])
        neg_penalty = float(probs[1:].max()) * 0.65
        return pos - neg_penalty

    def unload(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def safe_margin_crop_resize(img: Image.Image, target_w=WIDTH, target_h=HEIGHT, margin=LOGO_SAFE_MARGIN) -> Image.Image:
    """قص 8.5% من الحواف الأربعة لطرد لوجوهات زوايا القنوات قبل ضبط مقاس 16:9."""
    img = img.convert("RGB")
    w, h = img.size
    mx, my = int(w * margin), int(h * margin)
    if w - 2 * mx > 120 and h - 2 * my > 120:
        img = img.crop((mx, my, w - mx, h - my))
    w, h = img.size
    target_ratio = target_w / target_h
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        nw = int(h * target_ratio)
        left = (w - nw) // 2
        img = img.crop((left, 0, left + nw, h))
    else:
        nh = int(w / target_ratio)
        top = (h - nh) // 2
        img = img.crop((0, top, w, top + nh))
    return img.resize((target_w, target_h), Image.Resampling.BICUBIC)


def suppress_static_corner_logos(frames: list) -> list:
    """اكتشاف أي لوجو ثابت متبقٍ في الزوايا عبر الانحراف الزمني وتمويهه بنعومة."""
    if len(frames) < 8:
        return frames
    sample_idx = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
    stack = np.stack([np.asarray(frames[i], dtype=np.float32) for i in sample_idx], axis=0)
    H, W = stack.shape[1], stack.shape[2]
    cw, ch = int(W * 0.16), int(H * 0.13)
    corners = [
        (0, 0, cw, ch),
        (W - cw, 0, W, ch),
        (0, H - ch, cw, H),
        (W - cw, H - ch, W, H),
    ]
    active_boxes = []
    for (x1, y1, x2, y2) in corners:
        patch_seq = stack[:, y1:y2, x1:x2, :]
        temporal_std = patch_seq.std(axis=0).mean(axis=-1)
        spatial_contrast = patch_seq.mean(axis=0).std(axis=-1)
        static_logo_mask = (temporal_std < 3.2) & (spatial_contrast > 18.0)
        if static_logo_mask.mean() > 0.035:
            active_boxes.append((x1, y1, x2, y2))

    if not active_boxes:
        return frames

    cleaned = []
    for fr in frames:
        out = fr.copy()
        for (x1, y1, x2, y2) in active_boxes:
            crop = out.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=16))
            out.paste(crop, (x1, y1))
        cleaned.append(out)
    return cleaned


def _download_yt_clip(query: str, start_sec: int, duration: float, out_mp4: str) -> bool:
    end_sec = start_sec + max(4.5, duration + 1.2)
    cmd = [
        "yt-dlp", f"ytsearch3:{query}",
        "--match-filter", "duration > 25 & duration < 1800",
        "-f", "bestvideo[height<=1080][ext=mp4]/best[height<=1080][ext=mp4]/best",
        "--download-sections", f"*{start_sec}-{int(end_sec)}",
        "--max-downloads", "1", "--no-playlist", "--quiet", "--no-warnings",
        "-o", out_mp4,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=55)
    return os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 25_000


def _search_wikimedia_image(query: str, out_jpg: str) -> bool:
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6, "gsrlimit": 5, "prop": "imageinfo", "iiprop": "url",
        "iiurlwidth": 1920, "format": "json"
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15).json()
        pages = r.get("query", {}).get("pages", {})
        for p in pages.values():
            ii = p.get("imageinfo", [{}])[0]
            img_url = ii.get("thumburl") or ii.get("url")
            if img_url:
                ir = requests.get(img_url, headers=HEADERS, timeout=15)
                if ir.status_code == 200 and len(ir.content) > 15_000:
                    with open(out_jpg, "wb") as f:
                        f.write(ir.content)
                    return True
    except Exception:
        pass
    return False


def _probe_mid_frame(mp4_path: str) -> Image.Image:
    tmp_jpg = mp4_path + "_probe.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0.8", "-i", mp4_path, "-frames:v", "1", "-q:v", "3", tmp_jpg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if os.path.exists(tmp_jpg):
        im = Image.open(tmp_jpg).convert("RGB")
        os.remove(tmp_jpg)
        return safe_margin_crop_resize(im, 640, 360)
    return None


def fetch_single_shot_media(sh: dict, media_dir: str, gatekeeper: ClipGatekeeper) -> dict:
    sid = sh["id"]
    final_video = os.path.join(media_dir, f"shot_{sid:04d}.mp4")
    final_photo = os.path.join(media_dir, f"shot_{sid:04d}.jpg")

    if sh.get("asset_path") and os.path.exists(sh["asset_path"]):
        return sh

    vprompt = sh.get("visual_prompt", sh["spoken"])

    if sh["media_type"] == "video":
        best_path, best_score = None, -9.0
        for q_idx, q in enumerate(sh.get("yt_queries", [])[:3]):
            cand_mp4 = os.path.join(media_dir, f"cand_{sid:04d}_{q_idx}.mp4")
            try:
                if _download_yt_clip(q, int(sh.get("start_sec", 16)), sh["duration"], cand_mp4):
                    probe = _probe_mid_frame(cand_mp4)
                    if probe is not None:
                        sc = gatekeeper.score_image(probe, vprompt)
                        if sc > best_score:
                            best_score, best_path = sc, cand_mp4
                        if sc >= 0.35:
                            break
            except Exception:
                continue
        if best_path and os.path.exists(best_path):
            os.replace(best_path, final_video)
            sh["asset_path"] = final_video
            sh["clip_score"] = round(best_score, 3)
            # تنظيف باقي المرشحين
            for q_idx in range(3):
                c = os.path.join(media_dir, f"cand_{sid:04d}_{q_idx}.mp4")
                if os.path.exists(c):
                    os.remove(c)
            return sh

    # سحب الصور (لأزواج photo_in / photo_out أو كبديل طوارئ)
    for wq in sh.get("wiki_queries", []) + sh.get("yt_queries", [])[:1]:
        if _search_wikimedia_image(wq, final_photo):
            try:
                im = Image.open(final_photo).convert("RGB")
                sc = gatekeeper.score_image(safe_margin_crop_resize(im, 640, 360), vprompt)
                sh["asset_path"] = final_photo
                sh["clip_score"] = round(sc, 3)
                if sh["media_type"] == "video":
                    sh["media_type"] = "photo_in"
                return sh
            except Exception:
                pass

    # لوحة احتياطية داكنة تمنع توقف خط الإنتاج في أي ظرف
    fallback = Image.new("RGB", (WIDTH, HEIGHT), (14, 16, 22))
    fallback.save(final_photo, quality=92)
    sh["asset_path"] = final_photo
    sh["clip_score"] = 0.0

    # لو اللقطة dual_compare نسحب لقطة النصف الأيمن
    if sh.get("overlay") == "dual_compare" and sh.get("secondary_yt_query"):
        sec_jpg = os.path.join(media_dir, f"shot_{sid:04d}_sec.jpg")
        _search_wikimedia_image(sh["secondary_yt_query"], sec_jpg)
        if os.path.exists(sec_jpg):
            sh["secondary_asset"] = sec_jpg

    return sh


def hunt_all_media(shots: list, media_dir: str, workers: int = 4, progress_cb=None):
    os.makedirs(media_dir, exist_ok=True)
    gatekeeper = ClipGatekeeper()
    done_count = 0
    total = len(shots)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_single_shot_media, sh, media_dir, gatekeeper): sh for sh in shots}
        for fut in as_completed(futures):
            res = fut.result()
            done_count += 1
            if progress_cb:
                frac = 0.28 + 0.30 * (done_count / max(1, total))
                progress_cb(
                    frac,
                    f"صيد وفحص الميديا (CLIP + قص اللوجو): اللقطة {done_count}/{total} (تطابق {int(max(0, res.get('clip_score', 0))*100)}%)"
                )

    gatekeeper.unload()
    return shots
