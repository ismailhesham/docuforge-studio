#!/usr/bin/env bash
set -e

echo "=== [1/3] تثبيت FFmpeg والخطوط ومكتبات الصوت ==="
apt-get update -qq && apt-get install -y -qq ffmpeg espeak-ng fonts-dejavu-core git

echo "=== [2/3] تثبيت مكتبات Python والذكاء الاصطناعي ==="
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "=== [3/3] تشغيل استوديو DocuForge بالواجهة العربية ==="
echo "اضغط على الرابط العام (https://....gradio.live) الذي سيظهر بالأسفل لفتح الواجهة:"
python3 app.py
