import gradio as gr
from config import KOKORO_VOICES
from engine.pipeline import quick_swap_single_shot, run_full_production

RTL_CSS = """
.gradio-container { direction: rtl; text-align: right; font-family: 'Segoe UI', Tahoma, sans-serif; }
h1, h2, h3, label, p, span { text-align: right !important; }
"""


def launch_production_ui(
    api_key, project_name, input_mode_label, script_text, target_minutes,
    tts_label, kokoro_label, voice_speed, ref_voice,
    logo_img, logo_text, logo_opacity, logo_pos_label,
    bgm_file, resume_flag,
    progress=gr.Progress(track_tqdm=False)
):
    if not api_key or not api_key.strip():
        raise gr.Error("من فضلك أدخل مفتاح DeepSeek API أولاً.")
    if not script_text or not script_text.strip():
        raise gr.Error("من فضلك أدخل الاسكربت الإنجليزي أو موضوع الفيديو.")

    logs = []

    def cb(frac, msg):
        progress(frac, desc=msg)
        pct = int(round(frac * 100))
        logs.append(f"[{pct:3d}%] {msg}")

    in_mode = "script" if "اسكربت جاهز" in input_mode_label else "topic"
    tts_choice = "f5" if "استنساخ" in tts_label else "kokoro"
    k_voice = KOKORO_VOICES.get(kokoro_label, "am_michael")
    pos_code = "top_right" if "أعلى" in logo_pos_label else "bottom_right"

    cb(0.01, "بدء تهيئة مشروع الإنتاج الوثائقي...")
    final_mp4, srt_path, _ = run_full_production(
        project_name=project_name,
        api_key=api_key,
        script_or_topic=script_text,
        input_mode=in_mode,
        target_minutes=int(target_minutes),
        tts_engine_choice=tts_choice,
        kokoro_voice=k_voice,
        voice_speed=float(voice_speed),
        ref_voice_file=ref_voice,
        logo_img_file=logo_img,
        logo_text=logo_text,
        logo_opacity=float(logo_opacity) / 100.0,
        logo_pos=pos_code,
        bgm_file=bgm_file,
        resume_checkpoint=bool(resume_flag),
        progress_cb=cb,
    )
    return final_mp4, final_mp4, srt_path, "\n".join(logs)


def handle_quick_swap(project_name, shot_num, new_q, logo_img, logo_text, logo_opacity, logo_pos_label, bgm_file):
    pos_code = "top_right" if "أعلى" in logo_pos_label else "bottom_right"
    mp4 = quick_swap_single_shot(
        project_name=project_name,
        shot_number=int(shot_num),
        new_query=new_q,
        logo_img_file=logo_img,
        logo_text=logo_text,
        logo_opacity=float(logo_opacity) / 100.0,
        logo_pos=pos_code,
        bgm_file=bgm_file,
    )
    return mp4, f"تم استبدال اللقطة رقم {int(shot_num)} وإعادة تجميع الفيديو بنجاح!"


with gr.Blocks(css=RTL_CSS, theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate"), title="DocuForge AI Studio") as demo:
    gr.Markdown("# 🎬 استوديو DocuForge لإنتاج الوثائقيات الآلي (Full HD 1080p — حتى 30 دقيقة)")

    with gr.Row():
        with gr.Column(scale=6):
            with gr.Group():
                gr.Markdown("### ١. إعدادات المشروع والاسكربت (DeepSeek Director)")
                with gr.Row():
                    api_key_in = gr.Textbox(label="مفتاح DeepSeek API Key", type="password", placeholder="sk-...")
                    project_name_in = gr.Textbox(label="اسم المشروع (للحفظ والاستكمال التلقائي)", value="episode_01")
                input_mode_in = gr.Radio(
                    ["إدخال اسكربت جاهز (يحافظ على كلامك بالحرف)", "كتابة الاسكربت آلياً من فكرة أو عنوان"],
                    value="إدخال اسكربت جاهز (يحافظ على كلامك بالحرف)", label="وضع الإدخال"
                )
                target_min_in = gr.Slider(1, 30, value=8, step=1, label="المدة المستهدفة بالدقائق (عند التوليد الآلي)")
                script_in = gr.Textbox(lines=9, label="الاسكربت الإنجليزي الكامل (أو فكرة الفيديو)", placeholder="Paste your full English documentary script here...")
                resume_in = gr.Checkbox(value=True, label="تفعيل الحفظ والاستكمال التلقائي لو انقطع الاتصال (Auto-Resume)")

            with gr.Accordion("٢. إعدادات التعليق الصوتي مفتوح المصدر (Kokoro-82M & F5-TTS)", open=True):
                tts_mode_in = gr.Radio(
                    ["Kokoro-82M (راوي وثائقي فوري فائق السرعة)", "استنساخ صوت مخصص (F5-TTS Voice Cloning)"],
                    value="Kokoro-82M (راوي وثائقي فوري فائق السرعة)", label="محرك الصوت"
                )
                with gr.Row():
                    kokoro_voice_in = gr.Dropdown(list(KOKORO_VOICES.keys()), value=list(KOKORO_VOICES.keys())[0], label="صوت الراوي في Kokoro")
                    speed_in = gr.Slider(0.85, 1.20, value=1.0, step=0.05, label="سرعة الإلقاء")
                ref_voice_in = gr.Audio(type="filepath", label="ارفع عينة صوت الراوي المرجعية (10-15 ثانية WAV/MP3 عند اختيار الاستنساخ)")

            with gr.Accordion("٣. اللوجو الشفاف الخاص بك (الزاوية اليمين) والموسيقى الخلفية", open=False):
                with gr.Row():
                    logo_img_in = gr.File(file_types=["image"], type="filepath", label="رفع صورة لوجو شفاف (PNG اختياري)")
                    logo_text_in = gr.Textbox(label="أو اكتب اسم قناتك كنص شفاف (بدون صورة)", placeholder="مثال: TECH CHRONICLES")
                with gr.Row():
                    logo_opacity_in = gr.Slider(10, 80, value=30, step=5, label="درجة وضوح اللوجو الشفاف (%) — 30% هادئ وغير مؤثر")
                    logo_pos_in = gr.Radio(["أعلى اليمين", "أسفل اليمين"], value="أعلى اليمين", label="موقع اللوجو")
                bgm_in = gr.Audio(type="filepath", label="موسيقى خلفية مخصصة (اختياري — يتم خفض صوتها تلقائياً وقت كلام الراوي)")

            start_btn = gr.Button("🚀 ابدأ إنتاج ورندرة الفيديو الوثائقي", variant="primary", size="lg")

        with gr.Column(scale=5):
            gr.Markdown("### 📊 متابعة مراحل الإنتاج والمعاينة")
            status_log = gr.Textbox(lines=10, label="سجل التقدم اللحظي (من تقسيم الاسكربت حتى الرندرة النهائية)", interactive=False)
            video_out = gr.Video(label="معاينة الفيديو الوثائقي النهائي (1080p)")
            with gr.Row():
                mp4_file_out = gr.File(label="تحميل ملف الفيديو (MP4)")
                srt_file_out = gr.File(label="تحميل ملف الترجمة المتزامن ليوتيوب (.SRT)")

            with gr.Accordion("🔄 تعديل سريع للقطة واحدة بعد المعاينة (في 20 ثانية)", open=False):
                with gr.Row():
                    swap_num_in = gr.Number(value=1, precision=0, label="رقم اللقطة المراد تغييرها")
                    swap_query_in = gr.Textbox(label="جملة بحث يوتيوب الجديدة للقطة", placeholder="cinematic microchip wafer laser 4k")
                swap_btn = gr.Button("تبديل اللقطة وإعادة التجميع السريع")
                swap_msg = gr.Textbox(label="نتيجة التبديل", interactive=False)

    start_btn.click(
        fn=launch_production_ui,
        inputs=[
            api_key_in, project_name_in, input_mode_in, script_in, target_min_in,
            tts_mode_in, kokoro_voice_in, speed_in, ref_voice_in,
            logo_img_in, logo_text_in, logo_opacity_in, logo_pos_in,
            bgm_in, resume_in
        ],
        outputs=[video_out, mp4_file_out, srt_file_out, status_log]
    )

    swap_btn.click(
        fn=handle_quick_swap,
        inputs=[project_name_in, swap_num_in, swap_query_in, logo_img_in, logo_text_in, logo_opacity_in, logo_pos_in, bgm_in],
        outputs=[video_out, swap_msg]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
