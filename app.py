"""Multiple Image Analyzer - AI image classification with MobileNetV2 (ImageNet)."""
import io
import uuid
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="Multiple Image Analyzer", page_icon="🔍", layout="wide")

NAVY, TEAL, RED, INK = "#0B1F44", "#12B89A", "#D93636", "#5B6B88"
MAX_MB = 10
PAGES = ["Home", "Analyzer", "Results", "Dashboard", "Reports"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Plus Jakarta Sans',system-ui,sans-serif; }}
.stApp {{ background:#F6F8FC; }}
[data-testid="stSidebar"] {{ background:#fff; border-right:1px solid #E6EAF2; }}
h1,h2,h3 {{ color:{NAVY}; letter-spacing:-0.02em; }}
.hero {{ background:{NAVY}; color:#fff; border-radius:20px; padding:48px 44px; margin-bottom:24px; }}
.hero h1 {{ color:#fff; font-size:2.6rem; font-weight:800; margin:0 0 12px; }}
.hero p {{ color:#C9D3EA; font-size:1.05rem; max-width:640px; margin:0; }}
.card {{ background:#fff; border:1px solid #E6EAF2; border-radius:16px; padding:20px;
        box-shadow:0 2px 10px rgba(11,31,68,.06); height:100%; }}
.card h4 {{ color:{NAVY}; margin:0 0 6px; }} .card p {{ color:{INK}; margin:0; font-size:.93rem; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:8px; color:#fff; font-size:.8rem; font-weight:600; }}
.bar {{ height:7px; border-radius:4px; background:#E6EAF2; overflow:hidden; margin:4px 0 10px; }}
.bar > div {{ height:100%; background:{TEAL}; }}
.pred {{ display:flex; justify-content:space-between; font-size:.85rem; color:{INK}; }}
div.stButton > button, div.stDownloadButton > button {{ border-radius:10px; font-weight:600; }}
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{
    background:{TEAL}; border:none; color:#fff; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid #E6EAF2; border-radius:16px; padding:16px;
        box-shadow:0 2px 10px rgba(11,31,68,.06); }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model ----
@st.cache_resource(show_spinner="Loading MobileNetV2 (ImageNet weights)...")
def load_model():
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet", include_top=True)


def preprocess(raw: bytes) -> np.ndarray:
    """bytes -> RGB -> 224x224 -> float array (raises on corrupted files)."""
    img = Image.open(io.BytesIO(raw))
    img.load()
    img = img.convert("RGB").resize((224, 224))
    return np.asarray(img, dtype="float32")


def classify(items: list[dict]) -> list[dict]:
    """Batch-classify items; corrupted images are reported as failed, not dropped."""
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
    arrays, ok_idx, results = [], [], []
    for i, it in enumerate(items):
        base = {"id": it["id"], "name": it["name"], "size": it["size"], "bytes": it["bytes"]}
        try:
            arrays.append(preprocess(it["bytes"]))
            ok_idx.append(i)
            results.append({**base, "status": "Analyzed", "top3": []})
        except (UnidentifiedImageError, OSError, ValueError) as e:
            results.append({**base, "status": f"Failed: {type(e).__name__}", "top3": []})
    if arrays:
        batch = preprocess_input(np.stack(arrays))
        preds = load_model().predict(batch, verbose=0)
        for i, decoded in zip(ok_idx, decode_predictions(preds, top=3)):
            results[i]["top3"] = [(label.replace("_", " "), float(p) * 100) for _, label, p in decoded]
    return results


def quality(conf: float) -> tuple[str, str]:
    if conf >= 70:
        return "High confidence", TEAL
    if conf >= 40:
        return "Medium confidence", "#E6A100"
    return "Low confidence", RED


def fmt_size(n: int) -> str:
    return f"{n/1024:.1f} KB" if n < 1024**2 else f"{n/1024**2:.2f} MB"


# ---------------------------------------------------------------- state ----
ss = st.session_state
ss.setdefault("files", [])        # uploaded, not yet analyzed
ss.setdefault("results", [])      # latest analysis
ss.setdefault("history", [])      # session history of runs
ss.setdefault("uploader_key", 0)
ss.setdefault("page", "Home")
ss.setdefault("rejected", [])


def go(page: str):
    ss["page"] = page


def results_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        ok = bool(r["top3"])
        rows.append({
            "Filename": r["name"],
            "Predicted class": r["top3"][0][0] if ok else "-",
            "Confidence (%)": round(r["top3"][0][1], 2) if ok else None,
            "2nd prediction": f"{r['top3'][1][0]} ({r['top3'][1][1]:.1f}%)" if ok else "-",
            "3rd prediction": f"{r['top3'][2][0]} ({r['top3'][2][1]:.1f}%)" if ok else "-",
            "Analysis status": r["status"],
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.markdown(f"<h3 style='margin-bottom:0'>🔍 Image Analyzer</h3>"
                f"<p style='color:{INK};margin-top:2px'>Multiple Image Analyzer</p>", unsafe_allow_html=True)
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption(f"Queued: **{len(ss.files)}** · Analyzed: **{sum(bool(r['top3']) for r in ss.results)}**")
    if st.button("Reset analysis", use_container_width=True):
        ss.files, ss.results, ss.history, ss.rejected = [], [], [], []
        ss.uploader_key += 1
        st.toast("Analysis reset.")
        st.rerun()

page = ss.page

# ----------------------------------------------------------------- home ----
if page == "Home":
    st.markdown("""<div class="hero"><h1>Classify many images at once.</h1>
    <p>Upload a batch of photos and get the top three predicted categories with confidence scores,
    an analytics dashboard and a downloadable report. Powered by MobileNetV2 trained on ImageNet.</p></div>""",
                unsafe_allow_html=True)
    st.button("Start analyzing", type="primary", on_click=go, args=("Analyzer",))

    st.subheader("Key features")
    feats = [("Batch upload", "Drag and drop many JPG, JPEG or PNG files and preview them before analysis."),
             ("Top 3 predictions", "Every image gets three ranked labels with real model confidence."),
             ("Analytics", "Category distribution, average confidence and per-image comparison charts."),
             ("CSV report", "Download filenames, predicted classes, confidence and status.")]
    for col, (t, d) in zip(st.columns(4), feats):
        col.markdown(f"<div class='card'><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)

    st.subheader("How it works")
    steps = [("Upload", "Add your images."), ("Preprocess", "Convert to RGB, resize to 224×224, apply MobileNetV2 scaling."),
             ("Predict", "One batched forward pass through the network."), ("Review", "Browse results, charts and reports.")]
    for col, (t, d) in zip(st.columns(4), steps):
        col.markdown(f"<div class='card'><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)

    st.subheader("Technology")
    st.write("Python · Streamlit · TensorFlow/Keras · MobileNetV2 (ImageNet) · Pillow · NumPy · Pandas · Plotly")
    st.info("**Image classification, not object detection.** The model assigns one of 1,000 ImageNet "
            "categories to the whole image. It does not locate objects or draw boxes, and it can only "
            "name categories it was trained on.")

# ------------------------------------------------------------- analyzer ----
elif page == "Analyzer":
    st.title("Image analyzer")
    uploads = st.file_uploader("Drop images here or browse", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True, key=f"up_{ss.uploader_key}")
    if uploads:
        for f in uploads:
            data = f.getvalue()
            if len(data) > MAX_MB * 1024**2:
                ss.rejected.append(f"{f.name}: larger than {MAX_MB} MB")
                continue
            ss.files.append({"id": uuid.uuid4().hex, "name": f.name, "size": len(data), "bytes": data})
        ss.uploader_key += 1   # clear the widget so individual removal works
        st.rerun()
    for msg in ss.rejected:
        st.warning(f"Skipped {msg}")
    ss.rejected = []

    if not ss.files:
        st.info("No images yet. Upload at least one JPG, JPEG or PNG to begin.")
    else:
        c1, c2, _ = st.columns([1, 1, 4])
        analyze = c1.button("Analyze images", type="primary")
        if c2.button("Clear all"):
            ss.files = []
            st.rerun()

        cols = st.columns(4)
        for i, f in enumerate(list(ss.files)):
            with cols[i % 4]:
                try:
                    st.image(f["bytes"], use_container_width=True)
                except Exception:
                    st.error("Preview unavailable (file may be corrupted)")
                st.caption(f"**{f['name']}**  \n{fmt_size(f['size'])}")
                if st.button("Remove", key=f"rm_{f['id']}"):
                    ss.files = [x for x in ss.files if x["id"] != f["id"]]
                    st.rerun()

        if analyze:
            try:
                with st.spinner(f"Analyzing {len(ss.files)} image(s)..."):
                    ss.results = classify(ss.files)
                ss.history.append({"Run time": datetime.now().strftime("%H:%M:%S"),
                                   "Images": len(ss.results),
                                   "Analyzed": sum(bool(r["top3"]) for r in ss.results)})
                ss.files = []
                ss.page = "Results"
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# -------------------------------------------------------------- results ----
elif page == "Results":
    st.title("Prediction results")
    if not ss.results:
        st.info("Nothing to show yet. Upload and analyze images first.")
        st.button("Go to analyzer", on_click=go, args=("Analyzer",))
    else:
        cols = st.columns(3)
        for i, r in enumerate(ss.results):
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(r["bytes"], use_container_width=True)
                    st.markdown(f"**{r['name']}**")
                    if r["top3"]:
                        label, conf = r["top3"][0]
                        q, color = quality(conf)
                        st.markdown(f"<span class='badge' style='background:{color}'>{q}</span>", unsafe_allow_html=True)
                        st.markdown(f"### {label.title()} · {conf:.1f}%")
                        for lab, p in r["top3"]:
                            st.markdown(f"<div class='pred'><span>{lab}</span><span>{p:.1f}%</span></div>"
                                        f"<div class='bar'><div style='width:{min(p,100):.1f}%'></div></div>",
                                        unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='badge' style='background:{RED}'>{r['status']}</span>", unsafe_allow_html=True)
                        st.caption("This file could not be read as an image.")

# ------------------------------------------------------------ dashboard ----
elif page == "Dashboard":
    st.title("Analytics dashboard")
    if not ss.results:
        st.info("Analyze some images to see statistics.")
    else:
        df = results_df(ss.results)
        ok = df[df["Analysis status"] == "Analyzed"]
        m = st.columns(4)
        m[0].metric("Total images", len(df))
        m[1].metric("Successfully analyzed", len(ok))
        m[2].metric("Average confidence", f"{ok['Confidence (%)'].mean():.1f}%" if len(ok) else "-")
        m[3].metric("Most frequent category",
                    Counter(ok["Predicted class"]).most_common(1)[0][0] if len(ok) else "-")
        if len(ok):
            a, b = st.columns(2)
            dist = ok["Predicted class"].value_counts().reset_index()
            dist.columns = ["Category", "Images"]
            fig1 = px.bar(dist, x="Category", y="Images", title="Prediction category distribution",
                          color_discrete_sequence=[NAVY])
            a.plotly_chart(fig1, use_container_width=True)
            fig2 = px.bar(ok, x="Filename", y="Confidence (%)", title="Confidence by image",
                          color="Confidence (%)", color_continuous_scale=[RED, "#E6A100", TEAL],
                          range_color=[0, 100])
            b.plotly_chart(fig2, use_container_width=True)
        if ss.history:
            st.subheader("Session history")
            st.dataframe(pd.DataFrame(ss.history), use_container_width=True, hide_index=True)

# -------------------------------------------------------------- reports ----
elif page == "Reports":
    st.title("Reports")
    if not ss.results:
        st.info("Analyze images first, then download your report here.")
    else:
        df = results_df(ss.results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV report", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"image_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv", type="primary")
