import os

FPS = 30
WIDTH = 1920
HEIGHT = 1080

BG_DARK = (10, 11, 14, 255)
MUSTARD = (232, 185, 35, 255)
LOGO_SAFE_MARGIN = 0.085  # قص 8.5% من كل حافة لطرد لوجوهات المصادر

def _pick_font(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

FONT_BOLD = _pick_font([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
])
FONT_REG = _pick_font([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
}

KOKORO_VOICES = {
    "راوي وثائقي عميق - أمريكي (am_michael)": "am_michael",
    "راوي وثائقي سينمائي - أمريكي (am_adam)": "am_adam",
    "راوي وثائقي فخم - بريطاني (bm_george)": "bm_george",
    "راوي وثائقي هادئ - بريطاني (bm_lewis)": "bm_lewis",
    "راوية وثائقية - أمريكي (af_bella)": "af_bella",
}
