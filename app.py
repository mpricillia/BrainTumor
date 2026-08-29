import os
import json
import tempfile
import h5py

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras as K
from keras.models import Model
from keras.preprocessing.image import img_to_array

# ─── Auto-set light theme ─────────────────────────────────────────────────────
import pathlib
_cfg_dir = pathlib.Path(".streamlit"); _cfg_dir.mkdir(exist_ok=True)
_cfg_file = _cfg_dir / "config.toml"
if not _cfg_file.exists():
    _cfg_file.write_text("[theme]\nbase = \"light\"\nprimaryColor = \"#1a9896\"\nbackgroundColor = \"#f0f9f9\"\nsecondaryBackgroundColor = \"#e0f2f2\"\ntextColor = \"#0f2626\"\n")

st.set_page_config(
    page_title="NeuroScan",
    page_icon="🧠",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --teal:        #1a9896;
    --teal-dark:   #0d7070;
    --teal-light:  #e6f4f4;
    --teal-mid:    #b2dede;
    --white:       #f7f9f9;
    --grey-soft:   #eef2f2;
    --grey-text:   #6b8787;
    --dark:        #0f2626;
    --font-body:   'DM Sans', sans-serif;
    --font-mono:   'Space Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--white) !important;
    color: var(--dark) !important;
}

/* Force override Streamlit dark mode backgrounds */
.stApp, .stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], [data-testid="block-container"],
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
.main, .main > div, section[data-testid="stSidebar"],
div[data-testid="stToolbar"] {
    background-color: var(--white) !important;
    color: var(--dark) !important;
}

/* Streamlit text overrides */
p, span, label, div, h1, h2, h3, h4 {
    color: var(--dark) !important;
}

/* Override metric / info boxes */
[data-testid="stMetric"], [data-testid="stAlert"] {
    background-color: var(--teal-light) !important;
    color: var(--dark) !important;
}

/* Spinner color */
[data-testid="stSpinner"] * { color: var(--teal) !important; }

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 760px; }

/* ── Header ── */
.ns-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2.2rem;
    border-bottom: 2px solid var(--teal-mid);
    padding-bottom: 1.2rem;
}
.ns-logo {
    width: 48px; height: 48px;
    background: var(--teal);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.ns-title  { font-family: var(--font-mono); font-size: 1.55rem; font-weight: 700; color: var(--teal-dark); letter-spacing: -0.5px; }
.ns-sub    { font-size: 0.82rem; color: var(--grey-text); margin-top: 2px; font-weight: 400; }

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed var(--teal-mid);
    border-radius: 16px;
    background: var(--teal-light);
    padding: 2.2rem 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
    transition: border-color .2s;
}
.upload-zone:hover { border-color: var(--teal); }
.upload-icon { font-size: 2.4rem; margin-bottom: 0.5rem; }
.upload-label { font-size: 0.9rem; color: var(--grey-text); }

/* ── Streamlit file-uploader override ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] section {
    border: 2px dashed var(--teal-mid) !important;
    border-radius: 16px !important;
    background: var(--teal-light) !important;
    padding: 2rem 1.5rem !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--teal) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: var(--grey-text) !important;
    font-family: var(--font-body) !important;
}

/* ── Ganti foto button ── */
.stButton > button {
    background: transparent !important;
    color: var(--teal) !important;
    border: 1.5px solid var(--teal) !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.2rem !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: var(--teal) !important;
    color: white !important;
}

/* ── Image cards ── */
.img-card-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--teal-dark);
    margin-bottom: 0.45rem;
}

/* ── Result card ── */
.result-card {
    background: var(--teal);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    color: white;
}
.result-icon { font-size: 2rem; flex-shrink: 0; }
.result-label { font-size: 0.75rem; font-weight: 500; opacity: .75; text-transform: uppercase; letter-spacing: 1px; }
.result-class { font-family: var(--font-mono); font-size: 1.55rem; font-weight: 700; line-height: 1.15; }
.result-conf  { font-size: 0.85rem; opacity: .85; margin-top: 2px; }

/* ── Probability bars ── */
.prob-section { margin-top: 1.2rem; }
.prob-label   { font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
                text-transform: uppercase; color: var(--grey-text); margin-bottom: 0.8rem; }
.prob-row     { display: flex; align-items: center; gap: 10px; margin-bottom: 0.55rem; }
.prob-name    { font-size: 0.82rem; font-weight: 500; min-width: 90px; color: var(--dark); }
.prob-track   { flex: 1; height: 8px; background: var(--grey-soft); border-radius: 99px; overflow: hidden; }
.prob-fill    { height: 100%; border-radius: 99px; background: var(--teal); transition: width .6s ease; }
.prob-fill.top{ background: var(--teal-dark); }
.prob-pct     { font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; color: var(--dark); min-width: 46px; text-align: right; }

/* ── Divider ── */
.ns-divider { height: 1.5px; background: var(--grey-soft); margin: 1.2rem 0; border: none; }

/* ── Spinner text ── */
.stSpinner > div { border-top-color: var(--teal) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_DICT = {0: "glioma", 1: "meningioma", 2: "no tumor", 3: "pituitary"}
CLASS_ICON = {0: "🔴", 1: "🟠", 2: "🟢", 3: "🟡"}
MODEL_PATH = "modelf.h5"

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-header">
  <div class="ns-logo">🧠</div>
  <div>
    <div class="ns-title">NeuroScan</div>
    <div class="ns-sub">Brain Tumor MRI Classification · EfficientNetB1 + Grad-CAM</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Model helpers (same as before) ──────────────────────────────────────────
def build_model_architecture():
    base_model = K.applications.EfficientNetB1(weights=None, include_top=False, input_shape=(240, 240, 3))
    for layer in base_model.layers[:-50]: layer.trainable = False
    for layer in base_model.layers[-50:]: layer.trainable = True
    x = base_model.output
    x = K.layers.GlobalMaxPooling2D()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(256, activation="relu")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.3)(x)
    output = K.layers.Dense(4, activation="softmax")(x)
    return K.models.Model(inputs=base_model.input, outputs=output)

def patch_h5_model(original_path):
    patched_path = os.path.join(os.path.dirname(original_path), "modelf_patched_temp.h5")
    with h5py.File(original_path, "r") as src, h5py.File(patched_path, "w") as dst:
        for key in src.keys(): src.copy(key, dst)
        for key, value in src.attrs.items(): dst.attrs[key] = value
        model_config = src.attrs.get("model_config")
        if model_config is not None:
            if isinstance(model_config, bytes): model_config = model_config.decode("utf-8")
            config = json.loads(model_config)
            def fix_obj(obj):
                if isinstance(obj, dict):
                    if obj.get("class_name") == "InputLayer":
                        cfg = obj.get("config", {})
                        if "batch_shape" in cfg: cfg["batch_input_shape"] = cfg.pop("batch_shape")
                        cfg.pop("optional", None)
                    cfg = obj.get("config")
                    if isinstance(cfg, dict):
                        cfg.pop("optional", None); cfg.pop("quantization_config", None)
                        if "dtype" in cfg and isinstance(cfg["dtype"], dict):
                            dt = cfg["dtype"]
                            if dt.get("class_name") == "DTypePolicy": cfg["dtype"] = dt.get("config", {}).get("name", "float32")
                    obj.pop("quantization_config", None)
                    if "dtype" in obj and isinstance(obj["dtype"], dict):
                        dt = obj["dtype"]
                        if dt.get("class_name") == "DTypePolicy": obj["dtype"] = dt.get("config", {}).get("name", "float32")
                    for v in obj.values(): fix_obj(v)
                elif isinstance(obj, list):
                    for item in obj: fix_obj(item)
            fix_obj(config)
            dst.attrs["model_config"] = json.dumps(config).encode("utf-8")
    return patched_path

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    errors = []
    try: return K.models.load_model(MODEL_PATH, compile=False)
    except Exception as e: errors.append(f"Load langsung: {e}")
    try:
        patched = patch_h5_model(MODEL_PATH)
        return K.models.load_model(patched, compile=False)
    except Exception as e: errors.append(f"Patch config: {e}")
    try:
        model = build_model_architecture()
        try: model.load_weights(MODEL_PATH)
        except: model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=False)
        return model
    except Exception as e: errors.append(f"Rebuild + weights: {e}")
    st.error("Model gagal dimuat."); st.code("\n\n".join(errors)); return None

def crop_image(image):
    try:
        h, w = image.shape[:2]
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        _, t = cv2.threshold(g, 45, 255, cv2.THRESH_BINARY)
        t = cv2.erode(t, None, iterations=2); t = cv2.dilate(t, None, iterations=2)
        contours, _ = cv2.findContours(t.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: return image
        c = max(contours, key=cv2.contourArea)
        el = tuple(c[c[:,:,0].argmin()])[0]; er = tuple(c[c[:,:,0].argmax()])[0]
        et = tuple(c[c[:,:,1].argmin()])[0]; eb = tuple(c[c[:,:,1].argmax()])[0]
        # padding 5% supaya area pituitary (bawah) tidak kepotong
        pad = int(min(h, w) * 0.05)
        x1 = max(0, el[0] - pad); x2 = min(w, er[0] + pad)
        y1 = max(0, et[1] - pad); y2 = min(h, eb[1] + pad)
        cropped = image[y1:y2, x1:x2]
        return cropped if cropped.size > 0 else image
    except: return image

def predict_only(model, image):
    """Hanya prediksi tanpa GradCAM — untuk no tumor."""
    img = np.expand_dims(image, axis=0).astype("float32")
    preds = model.predict(img, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    return preds[0], pred_idx

def VizGradCAM(model, image, interpolant=0.35):
    # Ambil last Conv2D layer
    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
    target_layer    = model.get_layer(last_conv_layer.name)

    original_img   = image
    img            = np.expand_dims(original_img, axis=0).astype("float32")
    prediction     = model.predict(img, verbose=0)
    prediction_idx = int(np.argmax(prediction))

    with tf.GradientTape() as tape:
        gradient_model      = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, pred    = gradient_model(img)
        loss                = pred[:, prediction_idx]

    gradients      = tape.gradient(loss, conv2d_out)
    output         = conv2d_out[0]
    weights        = tf.reduce_mean(gradients[0], axis=(0, 1))

    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    activation_map = cv2.resize(activation_map.numpy(), (original_img.shape[1], original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    mn, mx = activation_map.min(), activation_map.max()
    if mx - mn > 0:
        activation_map = (activation_map - mn) / (mx - mn)
    activation_map = np.uint8(255 * activation_map)

    heatmap     = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    original_u8 = np.uint8((original_img - original_img.min()) / max(original_img.max() - original_img.min(), 1e-5) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    blended = np.uint8(original_u8 * interpolant + cvt_heatmap * (1 - interpolant))
    return blended, prediction[0], prediction_idx

# ─── Session state: track uploaded photo ─────────────────────────────────────
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = True
if "result" not in st.session_state:
    st.session_state.result = None

# ─── Load model ───────────────────────────────────────────────────────────────
model = load_model()
if model is None:
    st.error(f"❌ Model tidak ditemukan dari: `{MODEL_PATH}`")
    st.stop()

# ─── Upload / Ganti Foto ──────────────────────────────────────────────────────
if st.session_state.show_uploader:
    uploaded = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="mri_upload"
    )

    if uploaded is not None:
        # proses gambar
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Gambar gagal dibaca.")
            st.stop()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_cropped = crop_image(img_bgr)
        img_resized = cv2.resize(img_cropped, (240, 240))
        img_arr = img_to_array(img_resized)

        with st.spinner("Menganalisis..."):
            # prediksi dulu untuk cek kelas
            probs_pre, pred_idx_pre = predict_only(model, img_arr)
            # hanya jalankan GradCAM kalau ada tumor
            if pred_idx_pre != 2:  # 2 = no tumor
                gradcam_img, probs, pred_idx = VizGradCAM(model, img_arr)
            else:
                probs, pred_idx = probs_pre, pred_idx_pre
                gradcam_img = None

        st.session_state.result = {
            "img_rgb": img_rgb,
            "gradcam": gradcam_img,
            "probs": probs,
            "pred_idx": pred_idx,
        }
        st.session_state.show_uploader = False
        st.rerun()

else:
    r = st.session_state.result

    # ── Ganti foto button ──────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("↩ Ganti Foto"):
            st.session_state.show_uploader = True
            st.session_state.result = None
            st.rerun()

    st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)

    # ── Images side by side ───────────────────────────────────────────────
    has_heatmap = r["gradcam"] is not None
    col1, col2 = st.columns(2) if has_heatmap else (st.container(), None)

    if has_heatmap:
        with col1:
            st.markdown("<div class='img-card-label'>MRI Original</div>", unsafe_allow_html=True)
            st.image(r["img_rgb"], use_container_width=True)
        with col2:
            st.markdown("<div class='img-card-label'>Grad-CAM Heatmap</div>", unsafe_allow_html=True)
            st.image(r["gradcam"], use_container_width=True)
    else:
        with col1:
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                st.markdown("<div class='img-card-label' style='text-align:center'>MRI Original</div>", unsafe_allow_html=True)
                st.image(r["img_rgb"], use_container_width=True)
        st.markdown(
            "<div style='background:#e0f2f2;border-radius:10px;padding:0.75rem 1rem;"
            "font-size:0.82rem;color:#0d7070;margin-top:0.5rem'>"
            "ℹ️ <b>Grad-CAM tidak ditampilkan</b> — tidak ada area tumor yang perlu di-highlight.</div>",
            unsafe_allow_html=True
        )

    # ── Result card ───────────────────────────────────────────────────────
    pred_idx = r["pred_idx"]
    probs    = r["probs"]
    pred_cls = CLASS_DICT[pred_idx]
    pred_ico = CLASS_ICON[pred_idx]
    conf     = probs[pred_idx] * 100

    st.markdown(f"""
    <div class="result-card">
      <div class="result-icon">{pred_ico}</div>
      <div>
        <div class="result-label">Diagnosis</div>
        <div class="result-class">{pred_cls.title()}</div>
        <div class="result-conf">Confidence: {conf:.1f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bars ──────────────────────────────────────────────────
    bars_html = '<div class="prob-section"><div class="prob-label">Probability per class</div>'
    for idx, cls in CLASS_DICT.items():
        pct   = probs[idx] * 100
        top   = "top" if idx == pred_idx else ""
        bars_html += f"""
        <div class="prob-row">
          <div class="prob-name">{cls.title()}</div>
          <div class="prob-track"><div class="prob-fill {top}" style="width:{pct:.1f}%"></div></div>
          <div class="prob-pct">{pct:.1f}%</div>
        </div>"""
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)

    conf_val = r["probs"][r["pred_idx"]] * 100
    if conf_val < 70:
        st.markdown(
            f"<div style='background:#fff3cd;border-radius:10px;padding:0.75rem 1rem;"
            f"font-size:0.82rem;color:#856404;margin-top:0.5rem'>"
            f"⚠️ <b>Confidence rendah ({conf_val:.1f}%)</b> — model kurang yakin dengan prediksi ini. "
            f"Kemungkinan gambar kurang jelas atau kasus ambigu. Pertimbangkan pemeriksaan lanjutan.</div>",
            unsafe_allow_html=True
        )
    st.markdown("<br><small style='color:#6b8787;font-size:0.72rem'>⚠️ Hasil ini bukan diagnosis medis resmi. Konsultasikan dengan dokter spesialis.</small>", unsafe_allow_html=True)