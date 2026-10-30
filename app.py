"""Multiple Image Analyzer - Streamlit app (TensorFlow + MobileNetV2)."""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

MAX_IMAGES = 20

st.set_page_config(page_title="Multiple Image Analyzer", page_icon="🖼️", layout="wide")

st.markdown(
    """
    <style>
    .block-container{max-width:1120px;padding-top:2.2rem}
    .hero{text-align:center;padding:1.2rem 0 1.6rem}
    .pill{display:inline-block;padding:5px 14px;border:1px solid #e4e9f2;border-radius:999px;
          font-size:13px;font-weight:700;color:#0b2a6f;background:#fff}
    .hero h1{font-size:clamp(32px,5vw,56px);line-height:1.08;margin:16px 0 10px;color:#0f1b3d;
             letter-spacing:-.02em}
    .grad{background:linear-gradient(90deg,#0b2a6f,#2563eb 50%,#0ea5a4);
          -webkit-background-clip:text;background-clip:text;color:transparent}
    .hero p{max-width:620px;margin:0 auto;color:#55607a;font-size:18px}
    .stats{display:grid;grid-template-columns:repeat(4,1fr);max-width:900px;margin:28px auto 8px;
           background:#fff;border:1px solid #e4e9f2;border-radius:16px;
           box-shadow:0 10px 30px rgba(15,27,61,.07);overflow:hidden}
    .stats div{padding:16px 10px;text-align:center;border-right:1px solid #e4e9f2}
    .stats div:last-child{border-right:0}
    .stats b{display:block;color:#0b2a6f;font-size:16px}
    .stats span{font-size:12.5px;color:#55607a}
    .top{font-size:20px;font-weight:800;text-transform:capitalize;color:#0f1b3d;margin:4px 0 2px}
    .row{display:flex;justify-content:space-between;font-size:13px;color:#55607a;
         text-transform:capitalize;margin-top:8px}
    .track{height:7px;background:#f1f4fa;border-radius:8px;overflow:hidden;margin-top:3px}
    .track b{display:block;height:100%;background:linear-gradient(90deg,#0b2a6f,#2563eb,#0ea5a4)}
    @media(max-width:700px){.stats{grid-template-columns:1fr 1fr}}
    </style>
    <div class="hero">
      <span class="pill">AI-powered image classification</span>
      <h1>Multiple Image Analyzer<br><span class="grad">Analyze any image in bulk.</span></h1>
      <p>Upload a batch of photos and get the top five predicted labels with confidence
      scores for each one. Export everything as CSV.</p>
    </div>
    <div class="stats">
      <div><b>Any image</b><span>1,000 ImageNet classes</span></div>
      <div><b>Batch upload</b><span>Up to 20 per run</span></div>
      <div><b>Top 5 labels</b><span>With confidence</span></div>
      <div><b>CSV export</b><span>Download predictions</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MobileNetV2 model…")
def load_model():
    from tensorflow.keras.applications import MobileNetV2

    return MobileNetV2(weights="imagenet")


def predict(model, image: Image.Image):
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    x = preprocess_input(np.expand_dims(np.array(img, dtype="float32"), axis=0))
    preds = decode_predictions(model.predict(x, verbose=0), top=5)[0]
    return [(label.replace("_", " "), float(p) * 100) for _, label, p in preds]


def level(conf: float) -> str:
    return "High" if conf >= 70 else "Medium" if conf >= 40 else "Low"


st.markdown("### Upload images")
uploads = st.file_uploader(
    "Select multiple images · JPG, PNG, WebP",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if len(uploads) > MAX_IMAGES:
    st.warning(f"Only the first {MAX_IMAGES} images will be analyzed.")
    uploads = uploads[:MAX_IMAGES]

status = f"{len(uploads)} image(s) ready" if uploads else "No images selected yet."
st.caption(status + "  ·  Confidence: High 70%+ · Medium 40–69% · Low below 40%")

if st.button("Analyze images", type="primary", disabled=not uploads):
    model = load_model()
    results, bar = [], st.progress(0.0, text="Starting…")
    for i, f in enumerate(uploads):
        bar.progress(i / len(uploads), text=f"Analyzing {i + 1} of {len(uploads)}: {f.name}")
        try:
            f.seek(0)
            image = Image.open(f)
            labels = predict(model, image)
            results.append({"name": f.name, "bytes": f.getvalue(), "labels": labels})
        except Exception as e:  # unreadable / corrupt file
            results.append({"name": f.name, "bytes": f.getvalue(), "error": str(e)})
    bar.empty()
    st.session_state["results"] = results

results = st.session_state.get("results")
if results:
    ok = [r for r in results if "labels" in r]
    st.success(f"Done · {len(ok)} of {len(results)} analyzed")

    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3].container(border=True):
            st.image(r["bytes"], use_container_width=True)
            if "error" in r:
                st.error(f"{r['name']}: could not analyze this image.")
                continue
            top_label, top_conf = r["labels"][0]
            st.markdown(
                f'<div class="top">{top_label} · {top_conf:.0f}%</div>'
                f'<div style="font-size:12px;color:#55607a">{level(top_conf)} confidence</div>',
                unsafe_allow_html=True,
            )
            for label, conf in r["labels"]:
                st.markdown(
                    f'<div class="row"><span>{label}</span><span>{conf:.1f}%</span></div>'
                    f'<div class="track"><b style="width:{conf:.0f}%"></b></div>',
                    unsafe_allow_html=True,
                )

    if ok:
        rows = []
        for r in ok:
            row = {"file": r["name"], "top_label": r["labels"][0][0],
                   "confidence": round(r["labels"][0][1], 1)}
            for n, (label, conf) in enumerate(r["labels"], 1):
                row[f"label_{n}"] = f"{label} ({conf:.1f}%)"
            rows.append(row)
        st.download_button(
            "Download CSV",
            pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

st.divider()
st.caption("© 2026 Multiple Image Analyzer · Python · TensorFlow · MobileNetV2")
