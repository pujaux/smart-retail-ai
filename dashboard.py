import streamlit as st
import requests

API_URL = "https://smart-retail-ai-gu8u.onrender.com"
API_KEY = "retail-secret-key"
HEADERS = {"x-api-key": API_KEY}

st.set_page_config(page_title="Amber — Smart Retail AI", page_icon="🛍️", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; color: #1B2A41; }

.hero {
    padding: 2.4rem 2.6rem;
    background: linear-gradient(135deg, #1B2A41 0%, #2C4160 100%);
    border-radius: 18px;
    margin-bottom: 1.6rem;
}
.hero h1 { color: #F7F4EE !important; font-size: 2.5rem; margin: 0; }
.hero p { color: #C9924B; font-size: 1.05rem; margin-top: 0.5rem; font-weight: 500; }

.badge-row { display: flex; gap: 0.6rem; margin-top: 1rem; }
.badge {
    background: rgba(255,255,255,0.08);
    color: #F7F4EE;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.15);
}

div[data-testid="stTabs"] button { font-family: 'Sora', sans-serif; font-weight: 600; color: #1B2A41; }

.card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.6rem 1.7rem;
    box-shadow: 0 1px 3px rgba(27,42,65,0.08);
    border: 1px solid #ECE6D9;
    margin-bottom: 1rem;
}

.module-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    border: 1px solid #ECE6D9;
    height: 100%;
}
.module-card h4 { margin: 0 0 0.4rem 0; }
.module-card p { color: #5A6B85; font-size: 0.9rem; margin: 0; }
.module-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }

.result-chip {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.3rem;
    font-family: 'Sora', sans-serif;
}
.chip-positive { background: #E4F2E9; color: #2E7D5B; }
.chip-negative { background: #FBE7E5; color: #C0392B; }
.chip-neutral { background: #F0EDE6; color: #7A6A50; }
.chip-info { background: #EAF0F9; color: #2C4160; }

div.stButton > button {
    background-color: #C9924B;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
}
div.stButton > button:hover { background-color: #B57F3D; color: white; }

.footer {
    text-align: center;
    color: #8A93A3;
    font-size: 0.8rem;
    padding: 1.5rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>🛍️ Amber — Smart Retail AI Platform</h1>
    <p>Face recognition · Sentiment analysis · Product classification · Customer chatbot</p>
    <div class="badge-row">
        <span class="badge">FastAPI backend</span>
        <span class="badge">OpenCV</span>
        <span class="badge">scikit-learn</span>
        <span class="badge">DistilBERT</span>
        <span class="badge">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Overview", "👤  Face Recognition", "💬  Sentiment Analysis",
    "🤖  Chatbot", "📦  Product Classifier", "📊  Live Stats"
])

# ---------- Tab 0: Overview ----------
with tab0:
    st.markdown("#### What this platform does")
    c1, c2, c3, c4 = st.columns(4)
    modules = [
        ("👤", "Face Recognition", "Enrolls and recognizes returning customers using OpenCV LBPH."),
        ("💬", "Sentiment Analysis", "Classifies reviews as positive, negative, or neutral via TF-IDF + DistilBERT."),
        ("🤖", "Chatbot", "Answers common retail FAQs using intent matching."),
        ("📦", "Product Classifier", "Categorizes product photos (shoes, electronics, clothing)."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], modules):
        with col:
            st.markdown(f"""
            <div class="module-card">
                <div class="module-icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### How to use this dashboard")
    st.markdown("""
    Use the tabs above to try each module directly:
    - **Face Recognition** — enroll a customer photo, then upload a new photo to test recognition
    - **Sentiment Analysis** — paste any customer review to see its sentiment and confidence
    - **Chatbot** — chat naturally, ask about orders, returns, discounts, or payment methods
    - **Product Classifier** — upload a product photo to see its predicted category
    - **Live Stats** — see how many customers are enrolled and which modules are active
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 1: Face Recognition ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Enroll New Customer")
        name = st.text_input("Customer name", key="enroll_name", placeholder="e.g. Alice")
        enroll_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"], key="enroll_img")
        if enroll_file:
            st.image(enroll_file, width=180)
        if st.button("Enroll Customer", key="enroll_btn"):
            if name and enroll_file:
                files = {"file": (enroll_file.name, enroll_file.getvalue())}
                with st.spinner("Enrolling..."):
                    resp = requests.post(f"{API_URL}/enroll-customer", params={"name": name}, files=files, headers=HEADERS, timeout=60)
                if resp.status_code == 200:
                    st.success(f"Enrolled **{name}** successfully.")
                else:
                    st.error(resp.json())
            else:
                st.warning("Please provide both name and photo.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Recognize Customer")
        recog_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"], key="recog_img")
        if recog_file:
            st.image(recog_file, width=180)
        if st.button("Recognize", key="recog_btn"):
            if recog_file:
                files = {"file": (recog_file.name, recog_file.getvalue())}
                with st.spinner("Recognizing..."):
                    resp = requests.post(f"{API_URL}/recognize-face", files=files, headers=HEADERS, timeout=60)
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("match"):
                        st.markdown(f'<span class="result-chip chip-positive">✓ Welcome back, {result["name"]}!</span>', unsafe_allow_html=True)
                        st.caption(f"Confidence: {result['confidence']}")
                    else:
                        st.markdown('<span class="result-chip chip-info">Customer not recognized</span>', unsafe_allow_html=True)
                else:
                    st.error(resp.json())
            else:
                st.warning("Please upload a photo.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 2: Sentiment Analysis ----------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Analyze Customer Review Sentiment")
    review_text = st.text_area("Enter a customer review:", height=100, placeholder="e.g. The product quality is amazing...")
    use_bert = st.toggle("Use DistilBERT (slower, more accurate)", value=False)
    if st.button("Analyze Sentiment"):
        if review_text.strip():
            with st.spinner("Analyzing..."):
                resp = requests.post(f"{API_URL}/analyze-sentiment", json={"text": review_text, "use_distilbert": use_bert}, headers=HEADERS, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                sentiment = result["sentiment"]
                chip_class = {"positive": "chip-positive", "negative": "chip-negative", "neutral": "chip-neutral"}.get(sentiment, "chip-info")
                emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "")
                st.markdown(f'<span class="result-chip {chip_class}">{emoji} {sentiment.capitalize()}</span>', unsafe_allow_html=True)
                st.caption(f"Confidence: {result['confidence']*100:.1f}% · Model: {result.get('model', 'tfidf')}")
            else:
                st.error(resp.json())
        else:
            st.warning("Please enter a review.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 3: Chatbot ----------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Chat with Retail Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_msg = st.chat_input("Type your message...")
    if user_msg:
        st.session_state.chat_history.append(("user", user_msg))
        try:
            resp = requests.post(f"{API_URL}/chatbot", json={"message": user_msg}, headers=HEADERS, timeout=60)
            bot_reply = resp.json()["response"] if resp.status_code == 200 else "Sorry, something went wrong."
        except Exception:
            bot_reply = "Sorry, the server took too long to respond."
        st.session_state.chat_history.append(("assistant", bot_reply))
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 4: Product Classifier ----------
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Classify Product Image")
    st.caption("Lightweight classifier (color + texture features) — categories: shoes, electronics, clothing")
    prod_file = st.file_uploader("Upload product photo", type=["jpg", "jpeg", "png"], key="prod_img")
    if prod_file:
        st.image(prod_file, width=200)
    if st.button("Classify"):
        if prod_file:
            files = {"file": (prod_file.name, prod_file.getvalue())}
            with st.spinner("Classifying..."):
                resp = requests.post(f"{API_URL}/classify-product", files=files, headers=HEADERS, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                st.markdown(f'<span class="result-chip chip-info">{result["category"].capitalize()}</span>', unsafe_allow_html=True)
                st.caption(f"Confidence: {result['confidence']*100:.1f}%")
            else:
                st.error(resp.json())
        else:
            st.warning("Please upload a product photo.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 5: Live Stats ----------
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Platform Stats")
    if st.button("Refresh Stats"):
        with st.spinner("Loading..."):
            resp = requests.get(f"{API_URL}/dashboard/stats", headers=HEADERS, timeout=60)
        if resp.status_code == 200:
            stats = resp.json()
            c1, c2 = st.columns(2)
            c1.metric("Enrolled Customers", stats["enrolled_customers"])
            with c2:
                st.write("**Active Modules:**")
                for m in stats["modules_active"]:
                    st.write(f"✓ {m.replace('_', ' ').title()}")
        else:
            st.error(resp.json())
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Smart Retail AI Platform · FastAPI + Streamlit · Built as a 4-day capstone project</div>', unsafe_allow_html=True)
