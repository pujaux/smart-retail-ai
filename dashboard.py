import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"
API_KEY = "retail-secret-key"
HEADERS = {"x-api-key": API_KEY}

st.set_page_config(page_title="Amber — Smart Retail AI", page_icon="🛍️", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    color: #1B2A41;
}

.hero {
    padding: 2.2rem 2.5rem;
    background: linear-gradient(135deg, #1B2A41 0%, #2C4160 100%);
    border-radius: 18px;
    margin-bottom: 1.8rem;
}
.hero h1 {
    color: #F7F4EE !important;
    font-size: 2.4rem;
    margin: 0;
}
.hero p {
    color: #C9924B;
    font-size: 1.05rem;
    margin-top: 0.4rem;
    font-weight: 500;
}

div[data-testid="stTabs"] button {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    color: #1B2A41;
}

.card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 1px 3px rgba(27,42,65,0.08);
    border: 1px solid #ECE6D9;
    margin-bottom: 1rem;
}

div.stButton > button {
    background-color: #C9924B;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1.3rem;
}
div.stButton > button:hover {
    background-color: #B57F3D;
    color: white;
}

.stChatMessage {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>🛍️ Amber — Smart Retail AI Platform</h1>
    <p>Face recognition · Sentiment analysis · Customer chatbot, unified in one place</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤  Face Recognition", "💬  Sentiment Analysis", "🤖  Chatbot", "📊  Dashboard", "📦  Product Classifier"])

# ---------- Tab 1: Face Recognition ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Enroll New Customer")
        name = st.text_input("Customer name", key="enroll_name", placeholder="e.g. Alice")
        enroll_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"], key="enroll_img")
        if st.button("Enroll Customer", key="enroll_btn"):
            if name and enroll_file:
                files = {"file": (enroll_file.name, enroll_file.getvalue())}
                resp = requests.post(f"{API_URL}/enroll-customer", params={"name": name}, files=files, headers=HEADERS)
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
        if st.button("Recognize", key="recog_btn"):
            if recog_file:
                files = {"file": (recog_file.name, recog_file.getvalue())}
                resp = requests.post(f"{API_URL}/recognize-face", files=files, headers=HEADERS)
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("match"):
                        st.success(f"Welcome back, **{result['name']}**! (confidence: {result['confidence']})")
                    else:
                        st.info("Customer not recognized.")
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
    if st.button("Analyze Sentiment"):
        if review_text.strip():
            resp = requests.post(f"{API_URL}/analyze-sentiment", json={"text": review_text}, headers=HEADERS)
            if resp.status_code == 200:
                result = resp.json()
                sentiment = result["sentiment"]
                emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "")
                c1, c2 = st.columns(2)
                c1.metric("Sentiment", f"{emoji} {sentiment.capitalize()}")
                c2.metric("Confidence", f"{result['confidence']*100:.1f}%")
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
        resp = requests.post(f"{API_URL}/chatbot", json={"message": user_msg}, headers=HEADERS)
        bot_reply = resp.json()["response"] if resp.status_code == 200 else "Sorry, something went wrong."
        st.session_state.chat_history.append(("assistant", bot_reply))
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 4: Dashboard ----------
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Platform Stats")
    if st.button("Refresh Stats"):
        resp = requests.get(f"{API_URL}/dashboard/stats", headers=HEADERS)
        if resp.status_code == 200:
            stats = resp.json()
            c1, c2 = st.columns(2)
            c1.metric("Enrolled Customers", stats["enrolled_customers"])
            with c2:
                st.write("**Active Modules:**")
                for m in stats["modules_active"]:
                    st.write(f"✓ {m}")
        else:
            st.error(resp.json())
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 5: Product Classifier ----------
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Classify Product Image")
    st.caption("Lightweight classifier (color + texture features) — categories: shoes, electronics, clothing")
    prod_file = st.file_uploader("Upload product photo", type=["jpg", "jpeg", "png"], key="prod_img")
    if st.button("Classify"):
        if prod_file:
            files = {"file": (prod_file.name, prod_file.getvalue())}
            resp = requests.post(f"{API_URL}/classify-product", files=files, headers=HEADERS)
            if resp.status_code == 200:
                result = resp.json()
                c1, c2 = st.columns(2)
                c1.metric("Category", result["category"].capitalize())
                c2.metric("Confidence", f"{result['confidence']*100:.1f}%")
            else:
                st.error(resp.json())
        else:
            st.warning("Please upload a product photo.")
    st.markdown('</div>', unsafe_allow_html=True)
