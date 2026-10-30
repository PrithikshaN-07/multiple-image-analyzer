# Multiple Image Analyzer

Batch image classification with **MobileNetV2 (ImageNet)** in a single Streamlit app.
This is **image classification** (one label set per image), not object detection.

## Features
Multi-image upload (JPG/JPEG/PNG, 10 MB max each) · previews · remove/clear · top-3 predictions with
confidence · analytics dashboard (Plotly) · CSV report · reset · session history · corrupted-file handling.

## Run locally
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.10-3.12. First run downloads ~14 MB of ImageNet weights.

## Deploy free on Streamlit Community Cloud
1. Push this folder to a public GitHub repo (`git init`, `git add .`, `git commit`, `git push`).
2. Sign in at https://share.streamlit.io with GitHub -> **Create app** -> pick repo, branch `main`, file `app.py`.
3. Under **Advanced settings** choose Python 3.11 -> **Deploy**.
4. If the build runs out of memory, deploy the same files to a free CPU **Hugging Face Space** (SDK: Streamlit).

Free-tier limits: apps sleep after inactivity (wake on visit), ~1 GB RAM, shared CPU, no persistent storage
(uploads live only in the session).

## Pipeline
Upload -> RGB -> 224x224 -> NumPy -> `preprocess_input` -> batch -> `predict` -> `decode_predictions(top=3)`.
Model is loaded once via `st.cache_resource`. No training, no fake scores.

## Limitation
Only the 1,000 ImageNet categories can be predicted; e.g. most plant species will map to the nearest ImageNet class.
