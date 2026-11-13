"""Plant disease dashboard with MobileNetV2, Grad-CAM, and advisory."""

import hashlib
from html import escape
from io import BytesIO

from gtts import gTTS
import streamlit as st
from PIL import Image, UnidentifiedImageError

from decision_layer import JevDecisionLayer
from model_utils import PLANT_SPECIES, PlantDiseasePredictor

ALL_PLANTS_OPTION = "Auto-Detect Plant (All 38 Classes)"
PALETTE = {
    "canvas": "#EAE5D9",
    "surface": "#F2EDE2",
    "green": "#1E4638",
    "purple": "#5E4B8B",
    "text": "#182620",
    "muted": "#52615A",
    "shadow": "#D1CBBF",
}

st.set_page_config(
    page_title="Plant Disease XAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

      :root {
        --canvas: #EAE5D9;
        --surface: #F2EDE2;
        --green: #1E4638;
        --purple: #5E4B8B;
        --ink: #182620;
        --muted: #52615A;
        --shadow: #D1CBBF;
        --lift: 8px 8px 16px #D1CBBF, -8px -8px 16px #FFFFFF;
        --press: inset 5px 5px 10px #D1CBBF, inset -5px -5px 10px #FFFFFF;
        --radius: 16px;
      }

      html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--canvas);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
      }
      [data-testid="stMain"] > div {
        background: var(--canvas);
      }
      h1, h2, h3, [data-testid="stMetricLabel"] {
        color: var(--green) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.025em;
      }
      p, label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
        color: var(--ink);
      }
      [data-testid="stCaptionContainer"] p, small {
        color: var(--muted) !important;
      }
      [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        color: var(--green) !important;
        font-family: 'JetBrains Mono', monospace !important;
      }
      [data-testid="stMetric"] {
        color: var(--ink) !important;
        background: var(--surface) !important;
        border: 0 !important;
        border-radius: 16px !important;
        box-shadow: var(--lift) !important;
        padding: 14px 18px !important;
      }

      .bauhaus-header {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 180px;
        align-items: center;
        gap: 24px;
        padding: 18px 8px 20px;
        border-bottom: 2px solid var(--green);
        margin-bottom: 20px;
        color: var(--ink);
      }
      .bauhaus-kicker {
        margin: 0 0 5px;
        color: var(--purple);
        font: 700 0.72rem 'JetBrains Mono', monospace;
        letter-spacing: 0.14em;
      }
      .bauhaus-title {
        margin: 0;
        color: var(--green);
        font: 700 clamp(2rem, 5vw, 3.8rem)/0.98 'Space Grotesk', sans-serif;
        letter-spacing: -0.065em;
      }
      .bauhaus-subtitle {
        max-width: 720px;
        margin: 12px 0 0;
        color: var(--muted);
        font: 500 1rem/1.5 'DM Sans', sans-serif;
      }
      .bauhaus-motif {
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: 1fr 1fr;
        width: 140px;
        height: 110px;
        gap: 10px;
        justify-self: end;
        align-items: center;
        justify-items: center;
      }
      .motif-circle {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: var(--green);
      }
      .motif-square {
        width: 42px;
        height: 42px;
        background: var(--surface);
        border: 2px solid var(--green);
      }
      .motif-triangle {
        width: 48px;
        height: 42px;
        background: var(--purple);
        clip-path: polygon(50% 0, 0 100%, 100% 100%);
      }
      .motif-bar {
        width: 52px;
        height: 10px;
        background: var(--purple);
        transform: rotate(-18deg);
      }

      [class*="st-key-telemetry-"] [data-testid="stMetric"],
      .st-key-xai-card, .st-key-diagnosis-card, .st-key-advisory-card,
      .st-key-original-image-card, .st-key-gradcam-image-card,
      .st-key-gradcam-overlay-image-card,
      .st-key-scope-card, .st-key-input-well,
      .st-key-source_selector, .st-key-leaf_upload, .st-key-leaf_camera {
        color: var(--ink) !important;
        background: var(--surface) !important;
        border: 0 !important;
        border-radius: 16px !important;
        box-shadow: var(--lift) !important;
      }
      [class*="st-key-telemetry-"] [data-testid="stMetric"] {
        min-height: 102px;
      }
      .st-key-xai-card, .st-key-diagnosis-card, .st-key-advisory-card,
      .st-key-original-image-card, .st-key-gradcam-image-card, .st-key-scope-card {
        padding: 18px 20px;
      }
      .st-key-input-well {
        padding: 14px 18px;
        background: var(--canvas) !important;
        box-shadow: var(--press) !important;
      }
      .st-key-source_selector {
        padding: 10px 14px;
        background: var(--canvas) !important;
        box-shadow: var(--press) !important;
      }
      .st-key-leaf_upload, .st-key-leaf_camera {
        padding: 10px;
        background: var(--canvas) !important;
        box-shadow: var(--press) !important;
      }
      .st-key-scope-card {
        background: #E2DDD0 !important;
      }
      .xai-metrics-strip {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 12px 20px;
        margin-top: 16px;
        padding: 14px 16px;
        color: var(--ink);
        background: var(--surface);
        border-radius: var(--radius);
        box-shadow: var(--lift);
      }
      .xai-metric {
        display: flex;
        flex-direction: column;
        gap: 3px;
        min-width: 90px;
      }
      .xai-metric-label {
        color: var(--muted);
        font: 600 0.72rem 'DM Sans', sans-serif;
      }
      .xai-metric-value {
        color: var(--green);
        font: 600 0.9rem 'JetBrains Mono', monospace;
      }
      [data-testid="stSidebar"] {
        background: #E2DDD0;
        border-right: 2px solid var(--green);
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: var(--ink);
        background: var(--surface);
        border: 0;
        border-radius: 12px;
        box-shadow: var(--press);
      }
      [data-testid="stRadio"] label, [data-testid="stSelectbox"] label {
        color: var(--green) !important;
        font-weight: 700;
      }
      [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink);
      }
      [data-testid="stRadio"] input:checked + div {
        background-color: var(--green) !important;
        border-color: var(--green) !important;
      }
      [data-testid="stProgressBar"] > div > div {
        background-color: var(--green) !important;
      }
      [data-testid="stProgressBar"] > div {
        background-color: #D8D2C6 !important;
      }
      .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        color: #FFFFFF;
        background: var(--purple);
        border-radius: 999px;
        font: 600 0.82rem 'DM Sans', sans-serif;
      }
      .decision-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin: 8px 0 18px;
      }
      .decision-badge {
        padding: 7px 12px;
        color: #FFFFFF;
        background: var(--purple);
        border-radius: 999px;
        font: 600 0.82rem 'DM Sans', sans-serif;
      }
      .decision-badge.green {
        background: var(--green);
      }
      .section-mark {
        display: inline-block;
        margin-right: 9px;
        color: var(--purple);
        font: 700 0.75rem 'JetBrains Mono', monospace;
      }
      [data-testid="stExpander"] {
        border: 1px solid #D1CBBF;
        border-radius: 12px;
        background: var(--surface);
      }
      [data-testid="stExpander"] summary, [data-testid="stExpander"] p {
        color: var(--ink) !important;
      }
      @media (max-width: 700px) {
        .bauhaus-header { grid-template-columns: 1fr 92px; gap: 8px; }
        .bauhaus-motif { transform: scale(0.68); transform-origin: right center; }
      }
    </style>
    """
)


@st.cache_resource(show_spinner="Loading the trained MobileNetV2 checkpoint...")
def load_predictor() -> PlantDiseasePredictor:
    return PlantDiseasePredictor()


@st.cache_resource
def load_decision_layer() -> JevDecisionLayer:
    return JevDecisionLayer()


def readable_class_name(class_name: str) -> str:
    """Make PlantVillage folder labels easier to read without changing their IDs."""
    return " ".join(class_name.replace("___", " — ").replace("_", " ").split())


def readable_species_name(species: str) -> str:
    return species.replace("_", " ").replace("(", " (")


def render_badges(advisory: dict[str, str]) -> None:
    """Render controlled enum values as accessible, high-contrast status pills."""
    reliability = escape(advisory["reliability"].capitalize())
    severity = escape(advisory["severity"])
    routing = escape(advisory["routing"].replace("_", " ").capitalize())
    st.html(
        f"""
        <div class="decision-badges" role="group" aria-label="Decision status">
          <span class="decision-badge green">Reliability · {reliability}</span>
          <span class="decision-badge">Severity · {severity}</span>
          <span class="decision-badge green">Routing · {routing}</span>
        </div>
        """
    )


st.html(
    """
    <header class="bauhaus-header">
      <div>
        <p class="bauhaus-kicker">AGRICULTURAL DECISION SUPPORT · 38 CLASSES</p>
        <h1 class="bauhaus-title">PLANT DISEASE XAI</h1>
        <p class="bauhaus-subtitle">
          MobileNetV2 classification, visual focus inspection, and cautious
          structured guidance in one workspace.
        </p>
      </div>
      <div class="bauhaus-motif" aria-hidden="true">
        <span class="motif-circle"></span>
        <span class="motif-square"></span>
        <span class="motif-triangle"></span>
        <span class="motif-bar"></span>
      </div>
    </header>
    """
)
st.caption(
    "PlantVillage-trained predictions are decision-support only. They are not "
    "a confirmed diagnosis or a substitute for local agricultural expertise."
)

with st.sidebar:
    st.subheader("Plant species filter")
    selected_species = st.selectbox(
        "Choose a known species",
        options=[ALL_PLANTS_OPTION, *PLANT_SPECIES],
        format_func=lambda value: (
            ALL_PLANTS_OPTION
            if value == ALL_PLANTS_OPTION
            else readable_species_name(value)
        ),
        help=(
            "Optionally restrict and re-normalize the model's existing class "
            "probabilities to one species."
        ),
    )
    plant_filter = (
        None if selected_species == ALL_PLANTS_OPTION else selected_species
    )
    with st.container(key="scope-card"):
        st.markdown("**Model scope**")
        st.caption(
            "Trained on PlantVillage leaf images. Field conditions may differ; "
            "confirm important decisions with a local agricultural expert."
        )

try:
    predictor = load_predictor()
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(f"Could not load the trained checkpoint: {error}", icon="⚠️")
    st.stop()

decision_layer = load_decision_layer()

st.subheader("System telemetry")
telemetry_columns = st.columns(3)
telemetry_values = (
    ("Device", predictor.device.type.upper(), "telemetry-device"),
    ("Checkpoint", "Loaded", "telemetry-checkpoint"),
    ("Decision engine", decision_layer.mode, "telemetry-engine"),
)
for column, (label, value, key) in zip(telemetry_columns, telemetry_values):
    with column:
        with st.container(key=key):
            st.metric(label, value, border=True)
mode_pill = (
    "Live GenAI Decision Layer"
    if decision_layer.is_live
    else "Local Rule-Based Fallback"
)
st.html(
    f'<span class="status-pill" role="status">{escape(mode_pill)}</span>'
)

st.subheader("Image input")
with st.container(key="input-well"):
    input_source = st.radio(
        "Image source",
        ["Upload Leaf Image", "Capture via Camera"],
        horizontal=True,
        key="source_selector",
    )
    if input_source == "Upload Leaf Image":
        image_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, and PNG.",
            key="leaf_upload",
        )
    else:
        image_file = st.camera_input(
            "Take a photo of a leaf",
            help="Center one leaf in the frame and use good, even lighting.",
            key="leaf_camera",
        )

if image_file is None:
    st.info("Upload a leaf photo or capture one with your camera to begin.")
    st.stop()

try:
    image = Image.open(BytesIO(image_file.getvalue())).convert("RGB")
except (UnidentifiedImageError, OSError) as error:
    st.error(f"Could not read the selected image: {error}", icon="⚠️")
    st.stop()

with st.spinner("Running classification, Grad-CAM, and decision analysis..."):
    try:
        analysis = predictor.analyze(image, plant_filter=plant_filter)
        predictions = analysis["predictions"]
        top_prediction = predictions[0]
        evidence = {
            "predicted_class": top_prediction["class_name"],
            "confidence": float(top_prediction["confidence"]) / 100,
            "top_3": predictions,
            "gradcam_coverage": analysis["gradcam_coverage"],
            "plant_filter_used": plant_filter,
        }
        advisory = decision_layer.evaluate(evidence)
    except (RuntimeError, ValueError, TypeError) as error:
        st.error(f"Analysis failed: {error}", icon="⚠️")
        st.stop()

st.subheader("Visual inspection and diagnosis")
xai_column, diagnosis_column = st.columns([1.1, 0.9], gap="large")
with xai_column:
    with st.container(key="xai-card"):
        st.html(
            '<h3><span class="section-mark">03 / XAI</span>Visual evidence</h3>'
        )
        xai_metrics = analysis["xai_metrics"]
        image_columns = st.columns(3, gap="small")
        with image_columns[0]:
            with st.container(key="original-image-card"):
                st.image(
                    analysis["resized_image"],
                    caption="Original Leaf (224×224)",
                    alt="Leaf image resized to 224 by 224 for inference",
                    width="stretch",
                )
        with image_columns[1]:
            with st.container(key="gradcam-overlay-image-card"):
                st.image(
                    xai_metrics["raw_heatmap_img"],
                    caption="Raw Grad-CAM Heatmap",
                    alt="Standalone colorized raw Grad-CAM heatmap",
                    width="stretch",
                )
        with image_columns[2]:
            with st.container(key="gradcam-image-card"):
                st.image(
                    analysis["gradcam_overlay"],
                    caption="Grad-CAM Overlay",
                    alt="Grad-CAM heatmap blended with the leaf image",
                    width="stretch",
                )
        st.caption(
            "Red and warm regions indicate areas that influenced this class "
            "prediction; they are not a measurement of disease severity."
        )
        st.markdown(
            f"""
            <div class="xai-metrics-strip" role="group" aria-label="XAI evidence metrics">
              <div class="xai-metric"><span class="xai-metric-label">Method</span>
                <span class="xai-metric-value">{escape(str(xai_metrics['method']))}</span></div>
              <div class="xai-metric"><span class="xai-metric-label">Target Layer</span>
                <span class="xai-metric-value">features[-1]</span></div>
              <div class="xai-metric"><span class="xai-metric-label">Peak Activation</span>
                <span class="xai-metric-value">{float(xai_metrics['peak_activation']):.3f}</span></div>
              <div class="xai-metric"><span class="xai-metric-label">Coverage (&gt;0.5)</span>
                <span class="xai-metric-value">{float(xai_metrics['coverage_pct']):.1f}%</span></div>
              <div class="xai-metric"><span class="xai-metric-label">Leaf Focus Ratio</span>
                <span class="xai-metric-value">{float(xai_metrics['center_focus_pct']):.1f}%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with diagnosis_column:
    with st.container(key="diagnosis-card"):
        st.html(
            '<h3><span class="section-mark">04 / CLASSIFICATION</span>'
            "Primary diagnosis</h3>"
        )
        st.markdown(
            f"""
            <div style="color:var(--ink);background:var(--surface);
                        border-radius:var(--radius);box-shadow:var(--lift);
                        padding:14px 18px;margin-bottom:16px;">
              <div style="color:var(--muted);font-size:0.875rem;font-weight:600;">
                Top diagnosis
              </div>
              <div style="color:var(--green);font-size:1.35rem;line-height:1.25;
                          word-break:break-word;font-weight:700;margin-top:4px;">
                {escape(readable_class_name(str(top_prediction['class_name'])))}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        confidence = float(top_prediction["confidence"])
        st.metric("Confidence score", f"{confidence:.2f}%")
        st.progress(confidence / 100, text="Primary class probability")
        st.markdown("**Top three ranked classes**")
        for rank, prediction in enumerate(predictions, start=1):
            rank_confidence = float(prediction["confidence"])
            st.progress(
                rank_confidence / 100,
                text=f"{rank}. {readable_class_name(str(prediction['class_name']))} "
                f"— {rank_confidence:.2f}%",
            )
        if plant_filter:
            st.caption(
                f"Probabilities re-normalized within "
                f"{readable_species_name(plant_filter)} classes only."
            )

with st.container(key="advisory-card"):
    st.html(
        '<h3><span class="section-mark">05 / DECISION</span>'
        "Decision and advisory layer</h3>"
    )
    render_badges(advisory)
    st.html(
        f'<span class="status-pill" role="status">'
        f"{escape(advisory['engine_mode'])}</span>"
    )
    st.caption(advisory["status_reason"])
    st.markdown("**Symptom guidance**")
    st.write(advisory["issue_explanation"])
    st.markdown("**Recommended preliminary action**")
    st.write(advisory["recommended_action"])
    with st.expander("हिंदी सारांश और मार्गदर्शन"):
        hindi_summary = advisory["hindi_summary"]
        st.write(hindi_summary)
        audio_cache = st.session_state.setdefault("hindi_audio_cache", {})
        audio_key = hashlib.md5(hindi_summary.encode()).hexdigest()
        if audio_key not in audio_cache:
            try:
                audio_buffer = BytesIO()
                gTTS(text=hindi_summary, lang="hi", slow=False).write_to_fp(
                    audio_buffer
                )
                audio_cache[audio_key] = audio_buffer.getvalue()
            except Exception as error:
                st.caption(
                    "Hindi audio is unavailable offline "
                    f"({type(error).__name__})."
                )
        if audio_key in audio_cache:
            st.audio(audio_cache[audio_key], format="audio/mp3")
    st.caption(
        "Advisory text interprets the model evidence and never changes the "
        "MobileNetV2 predicted label. It is not a treatment prescription."
    )
