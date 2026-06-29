---
title: Trash Classifier SVM
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.49.0
app_file: app.py
pinned: false
---

# ♻️ Trash Classifier

Upload a photo of trash — get back **paper**, **glass**, or **plastic**.

A small computer-vision web app that classifies waste photos into three
recyclable categories using a Support Vector Machine on hand-crafted image
features (no deep learning, no GPU required).

Made by Group 10:
- 2802476424 - Vincensius Kevin Mulyono
- 2802536083 - Bintang Nur Fadhlillah
- 2802520860 - Clark Christopher Sompie
- 2802489206 - Malvin Raditya Nugraha
- 2802523515 - Muhammad Revi Alfarisi

**👉 Try it live:** https://huggingface.co/spaces/Gutstavo/TrashClassifier
** Watch it:** https://youtu.be/u_z5oxWu6WQ

## Results

Tested on a held-out set of 237 images (15% of the data):

| Metric | Score |
|---|---|
| Accuracy | **89.5%** |
| F1-score (macro) | **0.891** |
| Inference time | **~57 ms** (CPU) |

Trained on a balanced 3-class subset of [TrashNet](https://github.com/garythung/trashnet)
(594 paper, 501 glass, 482 plastic).

## Run it on your own machine

```bash
# 1. install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. launch the web app
streamlit run app.py
```

Then open http://localhost:8501 in your browser and upload an image.

> The trained model (`models/pipeline.joblib`) is included via Git LFS.
> If you cloned the repo without LFS, run `git lfs pull` to fetch it.

## Re-train the model

If you want to retrain from scratch:

1. Download the TrashNet `paper/`, `glass/`, and `plastic/` folders into the
   project root.
2. Open `train_model.ipynb` and run all cells. The notebook builds features,
   tunes the SVM with grouped 5-fold cross-validation, evaluates on the test
   set, and saves a fresh `models/pipeline.joblib`.

## Project files

```
app.py              Streamlit web app
train_model.ipynb   Training notebook (features → SVM → evaluate → save)
requirements.txt    Python dependencies
models/             Trained pipeline + reports (LFS)
```

## License & credits

Dataset: [TrashNet](https://github.com/garythung/trashnet) — Thung & Yang,
Stanford, 2016 (MIT License). Code in this repository is provided as-is for
educational use.
