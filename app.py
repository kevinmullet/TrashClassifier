# Trash classifier Streamlit app.
# Loads the SVM trained in train_model.ipynb and predicts on uploaded images.

import io
import os
import time

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import local_binary_pattern,graycomatrix,graycoprops,hog


MODELS_DIR = "models"
IMG_SIZE = 224
LABELS_ID = {"paper": "Kertas","glass": "Kaca","plastic": "Plastik"}


st.set_page_config(
    page_title="Trash Classifier (SVM)",
    page_icon="♻️",
    layout="centered",
)


# ---- feature extraction (must match what the notebook trained with) ----

def resize_img(img):
    return cv2.resize(img,(IMG_SIZE,IMG_SIZE))


def color_hist(img):
    feats = []
    for c in range(3):
        h = cv2.calcHist([img],[c],None,[32],[0,256]).flatten()
        h = h/(h.sum()+1e-7)
        feats.append(h)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    ranges = [(0,180),(0,256),(0,256)]
    for c,r in enumerate(ranges):
        h = cv2.calcHist([hsv],[c],None,[32],list(r)).flatten()
        h = h/(h.sum()+1e-7)
        feats.append(h)
    return np.concatenate(feats)


def color_moments(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    feats = []
    for src in (img,hsv):
        for c in range(3):
            ch = src[:,:,c].astype("float32")
            m = ch.mean()
            s = ch.std()+1e-7
            sk = float(np.mean(((ch-m)/s) ** 3))
            feats.extend([float(m)/255.0,float(s)/255.0,sk])
    return np.asarray(feats,dtype="float32")


def lbp_hist(gray):
    lbp = local_binary_pattern(gray,24,3,method="uniform")
    hist,_ = np.histogram(lbp.ravel(),bins=26,range=(0,26))
    hist = hist.astype("float32")
    hist /= (hist.sum()+1e-7)
    return hist


def glcm_feat(gray):
    g = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(
        g,
        distances=[1,2],
        angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    out = []
    for p in ["contrast","dissimilarity","homogeneity","energy","correlation"]:
        out.append(graycoprops(glcm,p).flatten())
    return np.concatenate(out).astype("float32")


def hog_feat(gray):
    # 32x32 cells -- must match hog_feat() in train_model.ipynb.
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(32,32),
        cells_per_block=(2,2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype("float32")


def edge_feat(gray):
    edges = cv2.Canny(gray,80,160)
    edge_ratio = float(edges.mean())/255.0
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag = np.sqrt(gx*gx+gy*gy)
    return np.asarray([
        edge_ratio,
        float(mag.mean())/255.0,
        float(mag.std())/255.0,
        float(np.percentile(mag,90))/255.0,
    ],dtype="float32")


def specular_feat(img):
    # Glass is transparent/reflective: bright,low-saturation specular
    # highlights. Plastic is opaque/more saturated. Targets the glass<->plastic
    # confusion. Must match specular_feat() in train_model.ipynb.
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1].astype("float32")
    v = hsv[:,:,2].astype("float32")
    spec = ((v > 230) & (s < 50)).astype("uint8")
    n_comp,_ = cv2.connectedComponents(spec)
    v_sorted = np.sort(v.ravel())
    top1 = v_sorted[int(0.99*v_sorted.size):]
    return np.asarray([
        float(spec.mean()),
        float((v > 240).mean()),
        float(top1.mean())/255.0,
        float(v.mean())/255.0,
        float(v.std())/255.0,
        float(s.mean())/255.0,
        float(s.std())/255.0,
        min(n_comp-1,200)/200.0,
    ],dtype="float32")


def region_color(img):
    # Spatial-pyramid HSV histogram (whole image+2x2 quadrants),compact
    # (H12,S12,V6) per region. Must match region_color() in train_model.ipynb.
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,W = hsv.shape[:2]
    hy,hx = H // 2,W // 2
    regions = [
        (0,0,H,W),
        (0,0,hy,hx),(0,hx,hy,W),(hy,0,H,hx),(hy,hx,H,W),
    ]
    chans = [(0,12,180),(1,12,256),(2,6,256)]
    feats = []
    for (y0,x0,y1,x1) in regions:
        sub = hsv[y0:y1,x0:x1]
        for c,bins,rng in chans:
            h = cv2.calcHist([sub],[c],None,[bins],[0,rng]).flatten()
            h = h/(h.sum()+1e-7)
            feats.append(h.astype("float32"))
    return np.concatenate(feats)


def extract_features(img):
    img = resize_img(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return np.concatenate([
        color_hist(img),
        color_moments(img),
        lbp_hist(gray),
        glcm_feat(gray),
        hog_feat(gray),
        edge_feat(gray),
        specular_feat(img),
        region_color(img),
    ]).astype("float32")


# ---- model loading+prediction ----

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # Single self-contained object: ColumnTransformer (per-block) -> SVC.
    pipeline = joblib.load(os.path.join(MODELS_DIR,"pipeline.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR,"label_encoder.joblib"))
    return pipeline,le


def predict(img_bgr,pipeline,le):
    feats = extract_features(img_bgr).reshape(1,-1)
    probs = pipeline.predict_proba(feats)[0]
    idx = int(np.argmax(probs))
    label = str(le.classes_[idx])
    prob_map = {str(c): float(probs[i]) for i,c in enumerate(le.classes_)}
    return label,float(probs[idx]),prob_map


def pil_to_bgr(img):
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)


# ---- streamlit UI ----

def main():
    st.title("♻️ Trash Classifier")
    st.caption(
        "SVM (RBF) — Paper/Glass/Plastic. "
        "Features: color histogram+color moments+LBP+GLCM+HOG+edge density+specular highlights+region color."
    )

    if not os.path.exists(os.path.join(MODELS_DIR,"pipeline.joblib")):
        st.error(
            "Model files not found in `models/`. "
            "Run `train_model.ipynb` first,then commit "
            "`models/pipeline.joblib` and `models/label_encoder.joblib`."
        )
        return

    pipeline,le = load_model()

    uploaded = st.file_uploader(
        "Upload an image of trash (jpg/png)",
        type=["jpg","jpeg","png"],
    )
    if uploaded is None:
        st.info("Upload an image to classify it.")
        return

    img = Image.open(io.BytesIO(uploaded.read()))
    bgr = pil_to_bgr(img)

    col_img,col_pred = st.columns([1,1])
    with col_img:
        st.image(img,caption=uploaded.name,width="stretch")

    with col_pred:
        t0 = time.perf_counter()
        label,conf,probs = predict(bgr,pipeline,le)
        latency_ms = (time.perf_counter()-t0)*1000

        label_id = LABELS_ID.get(label,label)
        st.metric(label="Prediction",value=f"{label_id} ({label})")
        st.metric(label="Confidence",value=f"{conf*100:.1f}%")
        st.caption(f"Inference time: {latency_ms:.1f} ms")

    st.subheader("Class probabilities")
    sorted_probs = dict(sorted(probs.items(),key=lambda kv: kv[1],reverse=True))
    for c,p in sorted_probs.items():
        st.write(f"**{LABELS_ID.get(c,c)}** ({c})")
        st.progress(float(p),text=f"{p*100:.1f}%")


if __name__ == "__main__":
    main()
