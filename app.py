import streamlit as st
from streamlit_lottie import st_lottie
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import plotly.express as px
import requests
import torch
from utils.preprocess import clean_text
import base64   

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="FakeGuard AI", page_icon="🛡️", layout="wide")

# ====================== BACKGROUND ======================

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64("assets/background.jpg")

# ====================== CLEAN CSS (NO THEME CONFLICT) ======================
st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bg_img}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}

    .main .block-container {{
        background: transparent !important;
    }}

    /* Sidebar fixed color */
    [data-testid="stSidebar"] {{
        background: rgba(15, 15, 35, 0.98) !important;
    }}
</style>
""", unsafe_allow_html=True)

# ====================== STATIC COLORS ======================
card_real = "linear-gradient(90deg, #00ff88, #00cc66)"
card_fake = "linear-gradient(90deg, #ff3366, #cc1122)"

st.markdown(f"""
<style>
    .title {{font-size: 3.9rem; font-weight: 900; background: linear-gradient(90deg, #00ffaa, #00ccff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}}
    .result-real {{background: {card_real}; color: black; padding: 32px; border-radius: 25px;
                  text-align: center; font-size: 2.9rem; font-weight: bold;}}
    .result-fake {{background: {card_fake}; color: white; padding: 32px; border-radius: 25px;
                  text-align: center; font-size: 2.9rem; font-weight: bold;}}
</style>
""", unsafe_allow_html=True)

# ====================== LOTTIE ======================
@st.cache_data
def load_lottie(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

lottie_thinking = load_lottie("https://assets1.lottiefiles.com/packages/lf20_yf4q5zq3.json")
lottie_success = load_lottie("https://assets9.lottiefiles.com/packages/lf20_2k5x4q.json")
lottie_fail = load_lottie("https://assets4.lottiefiles.com/packages/lf20_5zq5zq5z.json")
lottie_confetti = load_lottie("https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("assets/logo.png", width=180)
    st.title("🛡️ FakeGuard AI")
    st.markdown("**Advanced Transformer Fake News Detector**")

    st.info("**Model**: DistilBERT\n**Accuracy**: ~97.8%")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []

st.markdown('<h1 class="title">🛡️ FakeGuard AI</h1>', unsafe_allow_html=True)
st.markdown("### *Advanced AI with Cinematic Animations*")

col1, col2 = st.columns([3, 1])
with col1:
    news_text = st.text_area("Paste News Article or Headline", height=240, placeholder="Enter full article here...")
with col2:
    uploaded = st.file_uploader("📤 Upload .txt file", type=["txt"])
    if uploaded:
        news_text = uploaded.read().decode()

# ====================== MODEL ======================
@st.cache_resource
def get_classifier():
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    model.config.id2label = {0: "Real", 1: "Fake"}
    model.config.label2id = {"Real": 0, "Fake": 1}
    return pipeline("text-classification", model=model, tokenizer=tokenizer, 
                    return_token_type_ids=False, device=-1)
    
# Simple Summary Function
def get_short_summary(text):
    sentences = text.replace("?", ".").replace("!", ".").split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) <= 3:
        return " ".join(sentences)
    else:
        return ". ".join(sentences[:2]) + ". " + sentences[-1] + "."

# ====================== ANALYZE ======================
if st.button("🚀 ANALYZE WITH AI", type="primary", use_container_width=True):
    if not news_text or len(news_text.strip()) < 20:
        st.error("Please enter sufficient news content!")
    else:
        with st.spinner("🔍 Analyzing with Transformer..."):
            if lottie_thinking:
                st_lottie(lottie_thinking, height=220, key="thinking", quality="high")
            
            classifier = get_classifier()
            result = classifier(news_text[:512])[0]
            
            label = result['label']
            score = result['score']
            final_label = "Real" if label in ["Real", "LABEL_0", 0, "0"] else "Fake"
            confidence = score * 100

            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Prediction": "✅ REAL NEWS" if final_label == "Real" else "🚨 FAKE NEWS",
                "Confidence": f"{confidence:.1f}%",
                "Snippet": news_text[:100] + "..."
            })

        st.markdown("### 🎯 Detection Result")
        
        if final_label == "Real":
            if lottie_success: st_lottie(lottie_success, height=180, key="success_anim", loop=False)
            if lottie_confetti: st_lottie(lottie_confetti, height=320, key="confetti", loop=False)
            st.markdown(f'<div class="result-real">✅ THIS IS REAL NEWS</div>', unsafe_allow_html=True)
        else:
            if lottie_fail: st_lottie(lottie_fail, height=210, key="fail_anim", loop=True)
            st.markdown(f'<div class="result-fake">🚨 THIS IS FAKE NEWS</div>', unsafe_allow_html=True)
        
        st.metric("Confidence Score", f"{confidence:.1f}%")
        st.progress(confidence / 100)
        
        # ====================== NEW SHORT SUMMARY ======================
        summary = get_short_summary(news_text)
        st.markdown("### 📝 Short Summary of the Article")
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

      
        
        #=================Probability Breakdown =============================
        fig = px.bar(
            x=["Real", "Fake"],
            y=[confidence if final_label == "Real" else 100 - confidence,
               confidence if final_label == "Fake" else 100 - confidence],
            color=["Real", "Fake"],
            color_discrete_map={"Real": "#00ff88", "Fake": "#ff3366"},
            title="Probability Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📜 Recent Detections")
if "history" in st.session_state and st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No detections yet.")

st.markdown("**Made with ❤️ in NEPAL | Model : DistilBERT Transformers **")
st.caption("Developed and Trained by Krishna")