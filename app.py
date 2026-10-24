import os
import json
import numpy as np
import gradio as gr
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep app.py, the trained model, and class_names.json in the same folder.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "skin_lesion_model_finetuned.keras")
LABEL_PATH = os.path.join(PROJECT_DIR, "class_names.json")

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

MODEL = tf.keras.models.load_model(MODEL_PATH)

DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma-like lesions.",
    "bcc": "Basal cell carcinoma, a type of skin cancer.",
    "bkl": "Benign keratosis-like lesions, usually non-cancerous.",
    "df": "Dermatofibroma, generally a benign skin growth.",
    "mel": "Melanoma, a potentially serious type of skin cancer.",
    "nv": "Melanocytic nevi, commonly known as moles.",
    "vasc": "Vascular lesions involving blood vessels.",
}

CSS = """
:root {
    --app-bg: #101426;
    --panel-bg: #191e32;
    --panel-border: #373d59;
    --purple: #7567F8;
    --blue: #398CF5;
}
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
}
body, .gradio-container {
    background: var(--app-bg) !important;
}
#hero {
    text-align: center;
    padding: 30px 18px 26px;
    margin: 8px 0 22px;
    border: 1px solid #454b7b;
    border-radius: 24px;
    background: linear-gradient(135deg, #17152f 0%, #25205a 52%, #153a58 100%);
}
#hero .eyebrow {
    color: #bdb5ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 0 0 12px;
}
#hero h1 {
    margin: 0 0 12px;
    font-size: clamp(27px, 5vw, 43px);
    line-height: 1.15;
    font-weight: 850;
    letter-spacing: 1.1px;
    color: #ffffff;
}
#hero .subtitle {
    margin: 0;
    color: #d6d9f2;
    font-size: 14px;
    line-height: 1.65;
}
.panel {
    background: var(--panel-bg) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 18px !important;
    padding: 16px !important;
}
.section-title {
    margin: 0 0 12px !important;
}
#analyze-btn {
    width: 100%;
    min-height: 56px !important;
    border: 0 !important;
    border-radius: 14px !important;
    background: linear-gradient(90deg, #7567F8, #398CF5) !important;
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    letter-spacing: .5px;
    box-shadow: 0 6px 20px rgba(91, 91, 230, .25);
}
#analyze-btn:hover {
    filter: brightness(1.09);
}
#result-box textarea {
    line-height: 1.65 !important;
}
#footer-note {
    text-align: center;
    color: #aeb6d0;
    font-size: 12px;
    line-height: 1.8;
    padding: 18px 10px 6px;
}
"""

def predict(image):
    if image is None:
        return "Please upload an image first.", {}, None

    # Gradio provides a NumPy image in RGB format.
    image = tf.keras.utils.array_to_img(image).convert("RGB").resize((96, 96))
    image = np.asarray(image, dtype=np.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    scores = MODEL.predict(image, verbose=0)[0]
    ranking = np.argsort(scores)[::-1]
    top_class = CLASS_NAMES[int(ranking[0])]

    result = (
        f"Highest-scoring category: {top_class.upper()} ({top_class})\n\n"
        f"Educational description:\n{DESCRIPTIONS[top_class]}\n\n"
        "Important: this is an experimental model output, not a diagnosis. "
        "Model scores are not calibrated probabilities. Consult a qualified "
        "healthcare professional for assessment."
    )

    score_dict = {
        CLASS_NAMES[i]: float(scores[i])
        for i in range(len(CLASS_NAMES))
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = ranking[::-1]
    ax.barh(
        [CLASS_NAMES[i] for i in ordered],
        [float(scores[i]) for i in ordered],
        color="#7567F8",
        edgecolor="#398CF5",
        linewidth=0.5,
    )
    ax.set_xlabel("Model score", color="#e6e9f5")
    ax.set_title("Seven-category score comparison", color="#ffffff", pad=12)
    ax.set_xlim(0, 1)
    ax.set_facecolor("#191e32")
    fig.patch.set_facecolor("#191e32")
    ax.tick_params(colors="#e6e9f5")
    ax.xaxis.label.set_color("#e6e9f5")
    for spine in ax.spines.values():
        spine.set_color("#454b66")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()

    return result, score_dict, fig


with gr.Blocks(
    title="Skin Lesion Prototype",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <header id="hero">
        <p class="eyebrow">✦ MACHINE LEARNING · MEDICAL IMAGE ANALYSIS ✦</p>
        <h1>SKIN LESION PROTOTYPE</h1>
        <p class="subtitle">
            Deep-learning image classification powered by MobileNetV2<br>
            An educational project using the HAM10000 dataset
        </p>
    </header>
    """)

    with gr.Group(elem_classes=["panel"]):
        gr.Markdown("### 01 · Upload your image", elem_classes=["section-title"])
        image_input = gr.Image(
            type="numpy",
            label="Skin lesion image",
            height=320,
        )
        gr.Markdown(
            "Upload a clear image to explore the model's experimental output."
        )
        # The button intentionally appears directly below the image upload.
        predict_button = gr.Button(
            "✦  ANALYSE IMAGE",
            variant="primary",
            elem_id="analyze-btn",
        )

    gr.Markdown("## 02 · Analysis results")

    with gr.Group(elem_classes=["panel"]):
        result_output = gr.Textbox(
            label="Predicted category & educational description",
            lines=6,
            interactive=False,
            elem_id="result-box",
        )

    score_output = gr.Label(
        num_top_classes=7,
        label="Scores for all seven categories",
    )
    chart_output = gr.Plot(label="Score comparison")

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_output, score_output, chart_output],
    )

    gr.HTML("""
    <footer id="footer-note">
        <b>MEDICAL DISCLAIMER</b><br>
        This experimental educational prototype is not clinically validated.
        It must not be used to diagnose, rule out, or treat any condition.
        Model scores are not calibrated probabilities. Seek professional
        medical advice for concerning skin changes.<br><br>
        Built with Python · TensorFlow · MobileNetV2 · Gradio
    </footer>
    """)


    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port
    )
    
