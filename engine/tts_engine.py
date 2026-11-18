import gc
import os
import subprocess
import numpy as np
import soundfile as sf
import torch
from config import FPS


def _fmt_srt_time(sec: float) -> str:
    ms = int(round((sec % 1) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(shots: list, srt_path: str):
    t = 0.0
    lines = []
    for idx, sh in enumerate(shots, 1):
        dur = sh["n_frames"] / FPS
        t_end = t + dur
        lines.append(f"{idx}\n{_fmt_srt_time(t)} --> {_fmt_srt_time(t_end)}\n{sh['spoken'].strip()}\n")
        t = t_end
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def synthesize_all_shots(
    shots: list,
    audio_dir: str,
    engine_type: str = "kokoro",
    kokoro_voice: str = "am_michael",
    speed: float = 1.0,
    ref_audio_path: str = None,
    progress_cb=None,
):
    os.makedirs(audio_dir, exist_ok=True)
    sr_target = 24000

    if engine_type == "f5" and ref_audio_path and os.path.exists(ref_audio_path):
        from f5_tts.api import F5TTS
        f5_model = F5TTS()
        clean_ref = os.path.join(audio_dir, "_ref_clean.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", ref_audio_path, "-t", "14", "-ac", "1", "-ar", "24000", clean_ref],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for idx, sh in enumerate(shots):
            wav_path = os.path.join(audio_dir, f"shot_{sh['id']:04d}.wav")
            if not os.path.exists(wav_path):
                wav, sr, _ = f5_model.infer(ref_file=clean_ref, ref_text="", gen_text=sh["spoken"], speed=speed)
                sf.write(wav_path, wav, sr)
            info = sf.info(wav_path)
            sh["audio_path"] = wav_path
            sh["duration"] = max(2.1, info.duration + 0.12)
            sh["n_frames"] = int(round(sh["duration"] * FPS))
            if progress_cb:
                progress_cb(0.12 + 0.16 * ((idx + 1) / len(shots)), f"استنساخ الصوت (F5-TTS): اللقطة {idx+1}/{len(shots)}")
        del f5_model
    else:
        from kokoro import KPipeline
        lang_code = "b" if kokoro_voice.startswith("b") else "a"
        pipe = KPipeline(lang_code=lang_code)
        for idx, sh in enumerate(shots):
            wav_path = os.path.join(audio_dir, f"shot_{sh['id']:04d}.wav")
            if not os.path.exists(wav_path):
                pieces = []
                for _, _, audio in pipe(sh["spoken"], voice=kokoro_voice, speed=speed):
                    if audio is not None:
                        pieces.append(np.asarray(audio, dtype=np.float32))
                full_wav = np.concatenate(pieces) if pieces else np.zeros(int(sr_target * 2.2), dtype=np.float32)
                pad = np.zeros(int(sr_target * 0.12), dtype=np.float32)
                sf.write(wav_path, np.concatenate([full_wav, pad]), sr_target)
            info = sf.info(wav_path)
            sh["audio_path"] = wav_path
            sh["duration"] = max(2.0, info.duration)
            sh["n_frames"] = int(round(sh["duration"] * FPS))
            if progress_cb:
                progress_cb(0.12 + 0.16 * ((idx + 1) / len(shots)), f"توليد التعليق الصوتي (Kokoro): اللقطة {idx+1}/{len(shots)}")
        del pipe

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # تجميع شريط التعليق الصوتي المتصل بالمللي ثانية
    full_chunks = []
    for sh in shots:
        data, sr = sf.read(sh["audio_path"], dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        exact_samples = int(round((sh["n_frames"] / FPS) * sr))
        if len(data) < exact_samples:
            data = np.pad(data, (0, exact_samples - len(data)))
        else:
            data = data[:exact_samples]
        full_chunks.append(data)

    master_vo_path = os.path.join(audio_dir, "master_voiceover.wav")
    sf.write(master_vo_path, np.concatenate(full_chunks), sr)
    return master_vo_path
