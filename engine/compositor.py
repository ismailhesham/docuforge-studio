import math
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import BG_DARK, FONT_BOLD, FONT_REG, FPS, HEIGHT, MUSTARD, WIDTH
from engine.media_hunter import safe_margin_crop_resize, suppress_static_corner_logos

_FONT_CACHE = {}


def _font(size: int, bold: bool = True):
    key = (size, bold)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - pow(1.0 - x, 3)


def ease_in_out_sine(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return -(math.cos(math.pi * x) - 1.0) / 2.0


def build_vignette_mask(w=WIDTH, h=HEIGHT) -> Image.Image:
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    alpha = np.clip((dist - 0.55) * 155, 0, 115).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


VIGNETTE_CACHE = build_vignette_mask()


def build_combined_static_overlay(watermark_layer: Image.Image = None) -> Image.Image:
    """دمج طبقة الـ Vignette مع اللوجو الشفاف مرة واحدة فقط في بداية المشروع لمضاعفة سرعة الرندرة."""
    if watermark_layer is None:
        return VIGNETTE_CACHE
    return Image.alpha_composite(VIGNETTE_CACHE, watermark_layer)


def subpixel_smooth_zoom(img: Image.Image, scale: float, out_w=WIDTH, out_h=HEIGHT) -> Image.Image:
    """حركة كاميرا سينمائية بدون أي اهتزاز بكسل عبر مصفوفة التحويل AFFINE في خطوة واحدة مباشرة."""
    w, h = img.size
    fit_ratio = min(w / out_w, h / out_h)
    inv = fit_ratio / scale
    dx = (w - out_w * inv) * 0.5
    dy = (h - out_h * inv) * 0.5
    return img.transform(
        (out_w, out_h),
        Image.Transform.AFFINE,
        (inv, 0.0, dx, 0.0, inv, dy),
        resample=Image.Resampling.BICUBIC,
    )


def _decode_video_ram_pipe(asset_path: str, n_frames: int) -> list:
    """سحب فريمات الفيديو وقص حواف الـ 8.5% مباشرة في الرام عبر FFmpeg Pipe بدون أي ملفات صور مؤقتة على القرص."""
    over_w, over_h = int(WIDTH * 1.08), int(HEIGHT * 1.08)
    vf_chain = (
        f"fps={FPS},"
        f"crop=iw*0.83:ih*0.83:iw*0.085:ih*0.085,"
        f"scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
        f"crop={over_w}:{over_h}"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", asset_path,
        "-vf", vf_chain,
        "-frames:v", str(n_frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    raw_bytes = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    frame_sz = over_w * over_h * 3
    total_read = len(raw_bytes) // frame_sz
    frames = []
    for i in range(total_read):
        chunk = raw_bytes[i * frame_sz : (i + 1) * frame_sz]
        frames.append(Image.frombytes("RGB", (over_w, over_h), chunk))
    return frames


def decode_shot_base_frames(sh: dict, tmp_dir: str = "") -> list:
    n_frames = sh["n_frames"]
    asset = sh.get("asset_path", "")
    mt = sh.get("media_type", "video")

    if mt == "video" and asset.endswith(".mp4") and os.path.exists(asset):
        raw_pil = _decode_video_ram_pipe(asset, n_frames)
        if raw_pil:
            while len(raw_pil) < n_frames:
                raw_pil.append(raw_pil[-1].copy())
            raw_pil = suppress_static_corner_logos(raw_pil[:n_frames])
            out = []
            denom = max(1, n_frames - 1)
            for i, fr in enumerate(raw_pil):
                t = ease_in_out_sine(i / denom)
                out.append(subpixel_smooth_zoom(fr, 1.0 + 0.045 * t))
            return out

    # معالجة أزواج الصور المتعاكسة (photo_in / photo_out) بدقة فائقة وخطوة واحدة
    base_img = Image.open(asset).convert("RGB") if (asset and os.path.exists(asset)) else Image.new("RGB", (WIDTH, HEIGHT), (15, 17, 24))
    oversized = safe_margin_crop_resize(base_img, int(WIDTH * 1.18), int(HEIGHT * 1.18))
    out = []
    denom = max(1, n_frames - 1)
    for i in range(n_frames):
        t = ease_in_out_sine(i / denom)
        scale = (1.0 + 0.12 * t) if mt != "photo_out" else (1.12 - 0.12 * t)
        out.append(subpixel_smooth_zoom(oversized, scale))
    return out


def prepare_watermark_layer(logo_img_path: str = None, logo_text: str = "", opacity: float = 0.30, position: str = "top_right") -> Image.Image:
    """بناء طبقة اللوجو الشفاف أو النص في الزاوية اليمين داخل المنطقة الآمنة."""
    if not logo_img_path and not (logo_text and logo_text.strip()):
        return None
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    alpha_mult = max(0.05, min(1.0, float(opacity)))
    pad_x, pad_y = 58, 48

    if logo_img_path and os.path.exists(logo_img_path):
        wm = Image.open(logo_img_path).convert("RGBA")
        max_w = 210
        ratio = min(1.0, max_w / max(1, wm.width))
        wm = wm.resize((int(wm.width * ratio), int(wm.height * ratio)), Image.Resampling.LANCZOS)
        r, g, b, a = wm.split()
        a = a.point(lambda p: int(p * alpha_mult))
        wm = Image.merge("RGBA", (r, g, b, a))
        x = WIDTH - wm.width - pad_x
        y = pad_y if position == "top_right" else (HEIGHT - wm.height - pad_y)
        layer.paste(wm, (x, y), wm)
    else:
        txt = logo_text.strip()
        draw = ImageDraw.Draw(layer)
        fnt = _font(30, bold=True)
        bbox = draw.textbbox((0, 0), txt, font=fnt)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = WIDTH - tw - pad_x
        y = pad_y if position == "top_right" else (HEIGHT - th - pad_y)
        a_val = int(255 * alpha_mult)
        draw.text((x + 2, y + 2), txt, font=fnt, fill=(0, 0, 0, int(a_val * 0.55)))
        draw.text((x, y), txt, font=fnt, fill=(255, 255, 255, a_val))
    return layer


def apply_shot_graphics(
    frame_rgb: Image.Image,
    sh: dict,
    f_idx: int,
    watermark_layer: Image.Image = None,
    sec_img: Image.Image = None,
    static_overlay: Image.Image = None,
) -> Image.Image:
    overlay_type = sh.get("overlay", "clean")
    base_static = static_overlay if static_overlay is not None else build_combined_static_overlay(watermark_layer)
    canvas = frame_rgb.convert("RGBA")

    # مسار سريع فوري للقطات النظيفة (80% من الفيديو) بدون إنشاء طبقات جرافيك فارغة
    if overlay_type == "clean":
        return Image.alpha_composite(canvas, base_static).convert("RGB")

    n_frames = sh["n_frames"]
    prog = f_idx / max(1, n_frames - 1)
    enter = ease_out_cubic(min(1.0, f_idx / 14.0))
    canvas = Image.alpha_composite(canvas, VIGNETTE_CACHE)

    ui = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ui)

    if overlay_type == "mustard_split":
        panel_w = int(WIDTH * 0.43 * enter)
        if panel_w > 10:
            draw.rectangle([0, 0, panel_w, HEIGHT], fill=BG_DARK)
            draw.rectangle([panel_w - 8, 0, panel_w, HEIGHT], fill=MUSTARD)
            if enter > 0.55:
                headline = str(sh.get("headline", "KEY SHIFT")).upper()[:16]
                subtext = str(sh.get("subtext", "")).upper()[:32]
                draw.text((76, HEIGHT // 2 - 85), headline, font=_font(76, True), fill=MUSTARD)
                draw.text((78, HEIGHT // 2 + 18), subtext, font=_font(30, False), fill=(235, 238, 245, 255))

    elif overlay_type == "lower_pill":
        slide_x = int(-620 + 690 * enter)
        y0 = HEIGHT - 175
        draw.rounded_rectangle([slide_x, y0, slide_x + 580, y0 + 104], radius=14, fill=(10, 11, 14, 232))
        draw.rectangle([slide_x, y0 + 14, slide_x + 9, y0 + 90], fill=MUSTARD)
        title = str(sh.get("pill_title", "DOCUMENTARY ARCHIVE")).upper()[:28]
        sub = str(sh.get("pill_sub", ""))[:42]
        draw.text((slide_x + 28, y0 + 18), title, font=_font(30, True), fill=(255, 255, 255, 255))
        draw.text((slide_x + 28, y0 + 58), sub, font=_font(22, False), fill=MUSTARD)

    elif overlay_type == "dual_compare":
        mid = WIDTH // 2
        if sec_img is not None:
            right_crop = sec_img.crop((mid // 2, 0, mid // 2 + mid, HEIGHT)).convert("RGBA")
            canvas.paste(right_crop, (mid, 0))
        draw.rectangle([mid - 4, 0, mid + 4, HEIGHT], fill=MUSTARD)
        drop_y = int(-130 + 168 * enter)
        draw.rounded_rectangle([mid - 310, drop_y, mid + 310, drop_y + 76], radius=12, fill=BG_DARK)
        ctitle = str(sh.get("compare_title", "DIRECT COMPARISON")).upper()[:30]
        draw.text((mid - 270, drop_y + 20), ctitle, font=_font(30, True), fill=MUSTARD)
        l_lbl = str(sh.get("left_label", "LEGACY")).upper()[:20]
        r_lbl = str(sh.get("right_label", "MODERN")).upper()[:20]
        draw.rounded_rectangle([60, HEIGHT - 115, 410, HEIGHT - 55], radius=8, fill=(10, 11, 14, 220))
        draw.text((82, HEIGHT - 100), l_lbl, font=_font(26, True), fill=(255, 255, 255, 255))
        draw.rounded_rectangle([mid + 60, HEIGHT - 115, mid + 410, HEIGHT - 55], radius=8, fill=(10, 11, 14, 220))
        draw.text((mid + 82, HEIGHT - 100), r_lbl, font=_font(26, True), fill=MUSTARD)

    elif overlay_type == "rolling_stat":
        try:
            target_num = float(str(sh.get("stat_target", 100)).replace(",", ""))
        except Exception:
            target_num = 100.0
        cur_val = target_num * ease_out_cubic(min(1.0, prog * 1.45))
        num_txt = f"{int(round(cur_val)):,}" if target_num >= 10 else f"{cur_val:.1f}"
        suffix = str(sh.get("stat_suffix", "%"))
        lbl = str(sh.get("stat_label", "MEASURED IMPACT")).upper()[:34]
        bx0, by0, bx1, by1 = WIDTH // 2 - 360, HEIGHT // 2 - 135, WIDTH // 2 + 360, HEIGHT // 2 + 135
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=20, fill=(10, 11, 14, int(228 * enter)))
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=20, outline=MUSTARD, width=4)
        draw.text((bx0 + 65, by0 + 38), f"{num_txt}{suffix}", font=_font(96, True), fill=MUSTARD)
        draw.text((bx0 + 68, by0 + 168), lbl, font=_font(30, True), fill=(255, 255, 255, 255))

    canvas = Image.alpha_composite(canvas, ui)
    if watermark_layer is not None:
        canvas = Image.alpha_composite(canvas, watermark_layer)
    return canvas.convert("RGB")
