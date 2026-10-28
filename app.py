import streamlit as st
import joblib

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Sentiment Analyzer")
st.write("Analyze text as Positive, Negative, or Neutral using Machine Learning.")

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

user_text = st.text_area(
    "Enter your text:",
    placeholder="Example: I love this product!",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):

    if not user_text.strip():
        st.warning("Please enter some text first.")

    else:
        prediction = model.predict([user_text])[0]
        probabilities = model.predict_proba([user_text])[0]
        confidence = max(probabilities) * 100

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "negative":
            st.error("😞 Negative Sentiment")
        else:
            st.info("😐 Neutral Sentiment")

        st.metric("Model Confidence", f"{confidence:.2f}%")
        st.progress(int(confidence))

st.markdown("---")
st.caption("Built with Python, Scikit-learn, TF-IDF and Streamlit")
