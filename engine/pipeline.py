import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import soundfile as sf
from PIL import Image
from config import FPS, HEIGHT, PROJECTS_DIR, WIDTH
from engine.compositor import (
    apply_shot_graphics,
    build_combined_static_overlay,
    decode_shot_base_frames,
    prepare_watermark_layer,
)
from engine.director import build_storyboard
from engine.media_hunter import ClipGatekeeper, fetch_single_shot_media, hunt_all_media, safe_margin_crop_resize
from engine.tts_engine import export_srt, synthesize_all_shots


def _has_nvenc() -> bool:
    """اختبار حقيقي لتشغيل مشفر كارت الشاشة NVENC على السيرفر."""
    try:
        res = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                "-c:v", "h264_nvenc", "-f", "null", "-"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8
        )
        return res.returncode == 0
    except Exception:
        return False


USE_NVENC = _has_nvenc()
RENDER_WORKERS = min(4, max(2, (os.cpu_count() or 4) // 2))


def render_single_shot_segment(sh: dict, seg_path: str, tmp_dir: str, watermark_layer=None, static_overlay=None):
    """رندر لقطة واحدة بأقصى سرعة مباشرة عبر الذاكرة إلى NVENC/x264 بدون استهلاك رامات إضافية."""
    frames = decode_shot_base_frames(sh, tmp_dir)
    sec_img = None
    if sh.get("secondary_asset") and os.path.exists(sh["secondary_asset"]):
        sec_img = safe_margin_crop_resize(Image.open(sh["secondary_asset"]))

    if static_overlay is None:
        static_overlay = build_combined_static_overlay(watermark_layer)

    vcodec = ["-c:v", "h264_nvenc", "-preset", "p2", "-cq", "19"] if USE_NVENC else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-threads", "2"]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "-",
        *vcodec, "-pix_fmt", "yuv420p", seg_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for idx, base_fr in enumerate(frames):
        comp = apply_shot_graphics(
            base_fr, sh, idx,
            watermark_layer=watermark_layer,
            sec_img=sec_img,
            static_overlay=static_overlay,
        )
        proc.stdin.write(comp.tobytes())
    proc.stdin.close()
    proc.wait()
    del frames


def _generate_cinematic_drone_bed(duration_sec: float, out_wav: str):
    sr = 24000
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    drone = 0.35 * np.sin(2 * np.pi * 55.0 * t) + 0.20 * np.sin(2 * np.pi * 82.4 * t)
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)
    bed = (drone * pulse * 0.16).astype(np.float32)
    sf.write(out_wav, bed, sr)


def mux_final_video(segments: list, voice_wav: str, custom_bgm_path: str, out_mp4: str, work_dir: str):
    concat_txt = os.path.join(work_dir, "segments_list.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")

    silent_mp4 = os.path.join(work_dir, "video_silent.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", silent_mp4],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    vo_info = sf.info(voice_wav)
    if custom_bgm_path and os.path.exists(custom_bgm_path):
        bgm_src = custom_bgm_path
    else:
        bgm_src = os.path.join(work_dir, "auto_bgm.wav")
        _generate_cinematic_drone_bed(vo_info.duration + 2.0, bgm_src)

    filter_complex = (
        "[1:a]aresample=24000,volume=1.05[vo];"
        "[2:a]aresample=24000,aloop=loop=-1:size=2e+09,atrim=0:" + f"{vo_info.duration:.2f}" + ",volume=0.22[bgm];"
        "[bgm][vo]sidechaincompress=threshold=0.03:ratio=6:attack=35:release=420[ducked];"
        "[vo][ducked]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", silent_mp4, "-i", voice_wav, "-i", bgm_src,
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return out_mp4


def run_full_production(
    project_name: str,
    api_key: str,
    script_or_topic: str,
    input_mode: str,
    target_minutes: int,
    tts_engine_choice: str,
    kokoro_voice: str,
    voice_speed: float,
    ref_voice_file: str,
    logo_img_file: str,
    logo_text: str,
    logo_opacity: float,
    logo_pos: str,
    bgm_file: str,
    resume_checkpoint: bool,
    progress_cb=None,
):
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name.strip()) or "docu_project"
    pdir = os.path.join(PROJECTS_DIR, safe_slug)
    audio_dir = os.path.join(pdir, "audio")
    media_dir = os.path.join(pdir, "media")
    seg_dir = os.path.join(pdir, "segments")
    for d in (pdir, audio_dir, media_dir, seg_dir):
        os.makedirs(d, exist_ok=True)

    sb_path = os.path.join(pdir, "storyboard.json")

    # المرحلة 1: الإخراج وتقسيم المشاهد عبر DeepSeek
    if resume_checkpoint and os.path.exists(sb_path):
        with open(sb_path, "r", encoding="utf-8") as f:
            shots = json.load(f)
        if progress_cb:
            progress_cb(0.12, f"تم استرجاع التقسيم المحفوظ ({len(shots)} لقطة) بنجاح!")
    else:
        shots = build_storyboard(api_key, script_or_topic, mode=input_mode, target_minutes=target_minutes, progress_cb=progress_cb)
        with open(sb_path, "w", encoding="utf-8") as f:
            json.dump(shots, f, indent=2, ensure_ascii=False)

    # المرحلة 2: توليد الصوت مفتوح المصدر (يحدد المدة الحقيقية لكل لقطة تلقائياً من الاسكربت)
    master_vo = synthesize_all_shots(
        shots, audio_dir, engine_type=tts_engine_choice, kokoro_voice=kokoro_voice,
        speed=voice_speed, ref_audio_path=ref_voice_file, progress_cb=progress_cb
    )
    srt_path = os.path.join(pdir, f"{safe_slug}_captions.srt")
    export_srt(shots, srt_path)
    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2, ensure_ascii=False)

    # المرحلة 3: صيد الفيديوهات والصور وفحص CLIP وقص اللوجوهات
    shots = hunt_all_media(shots, media_dir, workers=4, progress_cb=progress_cb)
    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2, ensure_ascii=False)

    # المرحلة 4: الرندرة المتوازية فائقة السرعة (Multi-Worker Turbo Render + NVENC)
    wm_layer = prepare_watermark_layer(logo_img_file, logo_text, logo_opacity, logo_pos)
    combined_static = build_combined_static_overlay(wm_layer)
    total = len(shots)
    seg_files = [os.path.join(seg_dir, f"seg_{sh['id']:04d}.mp4") for sh in shots]

    pending_tasks = []
    done_count = 0
    for sh, seg_mp4 in zip(shots, seg_files):
        if resume_checkpoint and os.path.exists(seg_mp4) and os.path.getsize(seg_mp4) > 10_000:
            done_count += 1
        else:
            pending_tasks.append((sh, seg_mp4))

    enc_tag = f"GPU NVENC x{RENDER_WORKERS}" if USE_NVENC else f"Turbo CPU x{RENDER_WORKERS}"
    if progress_cb and done_count > 0:
        frac = 0.58 + 0.35 * (done_count / max(1, total))
        progress_cb(frac, f"تخطي {done_count} لقطة مرندرة مسبقاً، جاري رندر المتبقي ({enc_tag})...")

    if pending_tasks:
        with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
            futures = {
                pool.submit(render_single_shot_segment, sh, seg_mp4, seg_dir, wm_layer, combined_static): sh["id"]
                for sh, seg_mp4 in pending_tasks
            }
            for fut in as_completed(futures):
                fut.result()
                done_count += 1
                if progress_cb:
                    frac = 0.58 + 0.35 * (done_count / max(1, total))
                    progress_cb(frac, f"رندر المونتاج التيربو ({enc_tag}): اكتملت {done_count} من {total} لقطة")

    # المرحلة 5: الدمج النهائي والمكساج الصوتي
    if progress_cb:
        progress_cb(0.95, "جاري المكساج الصوتي الذكي وتصدير النسخة النهائية Full HD...")
    final_mp4 = os.path.join(pdir, f"{safe_slug}_master_1080p.mp4")
    mux_final_video(seg_files, master_vo, bgm_file, final_mp4, pdir)

    if progress_cb:
        progress_cb(1.0, f"اكتمل إنتاج الفيديو بنجاح! ({total} لقطة — {final_mp4})")
    return final_mp4, srt_path, sb_path


def quick_swap_single_shot(project_name: str, shot_number: int, new_query: str, logo_img_file=None, logo_text="", logo_opacity=0.30, logo_pos="top_right", bgm_file=None):
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name.strip()) or "docu_project"
    pdir = os.path.join(PROJECTS_DIR, safe_slug)
    sb_path = os.path.join(pdir, "storyboard.json")
    if not os.path.exists(sb_path):
        raise FileNotFoundError("المشروع غير موجود بعد، قم بإنتاج الفيديو أولاً.")

    with open(sb_path, "r", encoding="utf-8") as f:
        shots = json.load(f)

    target = next((s for s in shots if int(s["id"]) == int(shot_number)), None)
    if not target:
        raise ValueError(f"اللقطة رقم {shot_number} غير موجودة في المشروع.")

    target["asset_path"] = ""
    target["media_type"] = "video"
    target["yt_queries"] = [new_query.strip()] + target.get("yt_queries", [])
    target["visual_prompt"] = new_query.strip()

    gk = ClipGatekeeper()
    fetch_single_shot_media(target, os.path.join(pdir, "media"), gk)
    gk.unload()

    wm_layer = prepare_watermark_layer(logo_img_file, logo_text, logo_opacity, logo_pos)
    combined_static = build_combined_static_overlay(wm_layer)
    seg_dir = os.path.join(pdir, "segments")
    seg_mp4 = os.path.join(seg_dir, f"seg_{target['id']:04d}.mp4")
    render_single_shot_segment(target, seg_mp4, seg_dir, watermark_layer=wm_layer, static_overlay=combined_static)

    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2, ensure_ascii=False)

    seg_files = [os.path.join(seg_dir, f"seg_{s['id']:04d}.mp4") for s in shots]
    master_vo = os.path.join(pdir, "audio", "master_voiceover.wav")
    final_mp4 = os.path.join(pdir, f"{safe_slug}_master_1080p.mp4")
    mux_final_video(seg_files, master_vo, bgm_file, final_mp4, pdir)
    return final_mp4
