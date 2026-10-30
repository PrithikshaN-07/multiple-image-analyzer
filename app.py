"""
Image Analyzer - professional multi-image classifier
Python + TensorFlow + MobileNetV2 (ImageNet, 1000 classes) + Wikipedia descriptions
"""
import html
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps

BRAND = "Image Analyzer"  # change to your own brand name
MAX_IMAGES = 2            # only one or two images can be analyzed at a time

st.set_page_config(page_title=f"{BRAND} - identify anything in your photos", page_icon="🔍", layout="wide")

# ------------------------------------------------------------------
# STYLE
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{ --ink:#0B1220; --muted:#5B6577; --line:#E6E9F0; --blue:#2B59FF; --soft:#EEF2FF; }
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; color:var(--ink); }
.stApp { background:#fff; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { max-width:1000px; padding-top:1.2rem; }

.nav { display:flex; align-items:center; justify-content:space-between; padding:.4rem 0 1.2rem; }
.brand { display:flex; align-items:center; gap:.6rem; font-weight:700; font-size:1.1rem; }
.mark { width:30px; height:30px; border-radius:9px; background:linear-gradient(135deg,#2B59FF,#8B5CF6); }
.nav small { color:var(--muted); font-size:.9rem; }

.hero { border:1px solid var(--line); border-radius:28px; padding:3.2rem 3rem;
  background: radial-gradient(900px 300px at 85% -10%, rgba(139,92,246,.18), transparent 60%),
              radial-gradient(700px 300px at 0% 110%, rgba(43,89,255,.14), transparent 60%), #FAFBFF; }
.hero h1 { font-size:clamp(2.2rem,5vw,3.6rem); line-height:1.05; font-weight:800; letter-spacing:-.035em; margin:0 0 1rem; max-width:760px; }
.hero p { font-size:1.15rem; color:var(--muted); max-width:620px; line-height:1.6; margin:0; }

.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.4rem 0 2.4rem; }
.stat { border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem; background:#fff; }
.stat b { display:block; font-size:1.6rem; font-weight:800; letter-spacing:-.02em; }
.stat span { color:var(--muted); font-size:.92rem; }

h2.sec { font-size:1.6rem; font-weight:700; letter-spacing:-.02em; margin:2rem 0 .3rem; }
p.sub { color:var(--muted); margin:0 0 1rem; }

.how { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1rem; }
.how div { border:1px solid var(--line); border-radius:16px; padding:1.2rem; }
.how b { display:block; margin-bottom:.3rem; }
.how span { color:var(--muted); font-size:.95rem; line-height:1.5; }

[data-testid="stFileUploader"] section { border:1.5px dashed #AAB8FF; background:#F8FAFF; border-radius:18px; padding:1.8rem; }
[data-testid="stFileUploader"] button { background:var(--blue); color:#fff; border:0; border-radius:10px; font-weight:600; }

div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; border-color:var(--line); }
.stImage img { border-radius:12px; }
.chip { display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:.35rem; }
.c-Dog{background:#FFF1DB;color:#9A5B00} .c-Cat{background:#F1E8FF;color:#6B2FC9}
.c-Vehicle{background:#DDF6F0;color:#0A6B58} .c-Other{background:#EEF0F4;color:#46505F}
.c-low{background:#FFE9E7;color:#B42318}
.kicker { color:var(--muted); font-size:.82rem; margin-top:.7rem; }
.name { font-size:1.45rem; font-weight:800; letter-spacing:-.02em; text-transform:capitalize; line-height:1.15; }
.conf { color:var(--muted); font-size:.95rem; }
.row { display:flex; align-items:center; gap:.6rem; margin:.28rem 0; font-size:.9rem; }
.row .n { width:42%; text-transform:capitalize; }
.row .t { flex:1; height:8px; background:#EEF1F6; border-radius:99px; overflow:hidden; }
.row .t i { display:block; height:100%; background:linear-gradient(90deg,#2B59FF,#8B5CF6); border-radius:99px; }
.row .p { width:2.8rem; text-align:right; color:var(--muted); }
.about { color:#2A3345; font-size:.93rem; line-height:1.55; margin-top:.7rem; }
.about a { color:var(--blue); text-decoration:none; font-weight:600; }

.stDownloadButton button { background:var(--ink); color:#fff; border:0; border-radius:10px; font-weight:600; padding:.5rem 1.2rem; }
.foot { text-align:center; color:var(--muted); font-size:.9rem; margin:3rem 0 1rem; }
@media (max-width:800px){ .stats,.how{grid-template-columns:1fr 1fr} .hero{padding:2rem 1.4rem} }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the AI model (first run only)...")
def load_model():
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet")


VEHICLES = {
    "sports car", "convertible", "cab", "jeep", "limousine", "minivan", "racer", "beach wagon", "pickup",
    "model t", "police van", "ambulance", "minibus", "go-kart", "tow truck", "trailer truck", "fire engine",
    "garbage truck", "school bus", "moped", "motor scooter", "mountain bike", "bicycle-built-for-two",
    "tractor", "trolleybus", "golfcart", "passenger car", "streetcar", "recreational vehicle", "moving van",
}
BIG_CATS = {"lion", "tiger", "leopard", "jaguar", "cheetah", "cougar", "lynx", "snow leopard"}


def categorize(idx: int, label: str) -> str:
    """ImageNet ids 151-268 are all dog breeds, 281-285 are domestic cats."""
    l = label.lower()
    if 151 <= idx <= 268:
        return "Dog"
    if 281 <= idx <= 285 or l in BIG_CATS:
        return "Cat"
    if l in VEHICLES:
        return "Vehicle"
    return "Other"


@st.cache_data(show_spinner=False)
def classify(data: bytes, top_k: int = 5):
    import io
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
    x = preprocess_input(np.expand_dims(np.array(img, dtype="float32"), 0))
    preds = load_model().predict(x, verbose=0)
    order = np.argsort(preds[0])[::-1][:top_k]
    named = decode_predictions(preds, top=top_k)[0]
    out = []
    for idx, (_, label, p) in zip(order, named):
        label = label.replace("_", " ")
        out.append({"label": label, "p": float(p), "cat": categorize(int(idx), label)})
    return out


@st.cache_data(show_spinner=False, ttl=86400)
def describe(label: str, cat: str):
    query = {"Dog": f"{label} dog breed", "Cat": f"{label} cat"}.get(cat, label)
    try:
        hit = requests.get("https://en.wikipedia.org/w/api.php",
                           params={"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"},
                           timeout=6).json()
        title = hit["query"]["search"][0]["title"]
        s = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                         timeout=6, headers={"User-Agent": "ImageAnalyzer/2.0"}).json()
        return s.get("extract", ""), s.get("content_urls", {}).get("desktop", {}).get("page", "")
    except Exception:
        return "", ""


# ------------------------------------------------------------------
# PAGE
# ------------------------------------------------------------------
st.markdown(
    f"""
<div class="nav"><div class="brand"><div class="mark"></div>{BRAND}</div><small>Free to use. No sign-up.</small></div>
<div class="hero">
  <h1>Know exactly what is in every photo.</h1>
  <p>Upload dogs, cats, cars or anything else. Get the breed or type, a confidence score and a short description for each image.</p>
</div>
<div class="stats">
  <div class="stat"><b>1,000</b><span>object categories</span></div>
  <div class="stat"><b>120</b><span>dog breeds</span></div>
  <div class="stat"><b>1 or 2</b><span>images per analysis</span></div>
  <div class="stat"><b>CSV</b><span>export your results</span></div>
</div>
<h2 class="sec">How it works</h2>
<p class="sub">A convolutional neural network learns visual patterns, so it needs no training from you.</p>
<div class="how">
  <div><b>1. Upload</b><span>Add one photo or a whole batch in JPG, PNG or WebP.</span></div>
  <div><b>2. Analyze</b><span>MobileNetV2 compares shapes, textures and colors against what it learned from millions of images.</span></div>
  <div><b>3. Understand</b><span>See the top predictions with scores, plus a plain-language description of the top match.</span></div>
</div>
<h2 class="sec">Analyze your images</h2>
<p class="sub">Upload one or two photos.</p>
""",
    unsafe_allow_html=True,
)

files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                         accept_multiple_files=True, label_visibility="collapsed")

if files and len(files) > MAX_IMAGES:
    st.warning(f"You can analyze up to {MAX_IMAGES} images at a time. Only the first {MAX_IMAGES} will be used.")
    files = files[:MAX_IMAGES]

if files:
    results = []
    bar = st.progress(0.0, text="Analyzing...")
    for i, f in enumerate(files):
        data = f.getvalue()
        results.append({"name": f.name, "data": data, "preds": classify(data)})
        bar.progress((i + 1) / len(files), text=f"Analyzed {i + 1} of {len(files)}")
    bar.empty()

    # ---- summary ----
    cats = [r["preds"][0]["cat"] for r in results]
    counts = {c: cats.count(c) for c in ["Dog", "Cat", "Vehicle", "Other"] if cats.count(c)}
    summary = "".join(f'<span class="chip c-{c}">{n} {c.lower()}{"s" if n > 1 else ""}</span>' for c, n in counts.items())
    st.markdown(f'<h2 class="sec">Results</h2><p class="sub">{len(results)} images analyzed &nbsp; {summary}</p>',
                unsafe_allow_html=True)

    shown = results

    # ---- grid ----
    cols = st.columns(len(shown), gap="large")
    if True:
        for col, r in zip(cols, shown):
            top = r["preds"][0]
            text, url = describe(top["label"], top["cat"])
            with col, st.container(border=True):
                st.image(r["data"], use_container_width=True)
                kind = {"Dog": "Breed", "Cat": "Type", "Vehicle": "Vehicle type"}.get(top["cat"], "Identified as")
                low = '<span class="chip c-low">Low confidence</span>' if top["p"] < 0.35 else ""
                st.markdown(
                    f'<span class="chip c-{top["cat"]}">{top["cat"]}</span>{low}'
                    f'<div class="kicker">{kind}</div>'
                    f'<div class="name">{html.escape(top["label"])}</div>'
                    f'<div class="conf">{top["p"] * 100:.0f}% confidence</div>',
                    unsafe_allow_html=True,
                )
                rows = "".join(
                    f'<div class="row"><span class="n">{html.escape(p["label"])}</span>'
                    f'<span class="t"><i style="width:{p["p"] * 100:.1f}%"></i></span>'
                    f'<span class="p">{p["p"] * 100:.0f}%</span></div>' for p in r["preds"][:3])
                about = (f'<div class="about">{html.escape(text[:260])}{"..." if len(text) > 260 else ""} '
                         f'<a href="{html.escape(url)}" target="_blank">Read more</a></div>') if text else ""
                st.markdown(f'<div style="margin-top:.6rem">{rows}</div>{about}', unsafe_allow_html=True)

    # ---- export ----
    df = pd.DataFrame([{"file": r["name"], "category": r["preds"][0]["cat"], "prediction": r["preds"][0]["label"],
                        "confidence_%": round(r["preds"][0]["p"] * 100, 1)} for r in results])
    st.write("")
    st.download_button("Download results (CSV)", df.to_csv(index=False).encode(), "predictions.csv", "text/csv")
    st.caption("Dogs are matched to 120 breeds. Cats are matched to 5 types (tabby, tiger cat, Persian, Siamese, Egyptian) "
               "plus big cats. Cars are matched to body type, such as sports car, convertible or jeep, not brand or model.")

st.markdown(f'<div class="foot">{BRAND} · Built with Python, TensorFlow and MobileNetV2</div>', unsafe_allow_html=True)
