import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# Header
st.title("📩 Spam Message Detector")
st.write(
    "Use machine learning to determine whether an SMS message "
    "is likely to be spam."
)

st.divider()

# Message input
message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize!",
    height=150
)

# Prediction
if st.button("🔍 Check Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        # Convert text into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_tfidf)[0]

        # Probabilities
        probabilities = model.predict_proba(message_tfidf)[0]

        ham_probability = probabilities[0]
        spam_probability = probabilities[1]

        st.divider()

        # Result
        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")

        # Probability
        st.subheader("Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam Probability",
                f"{spam_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Ham Probability",
                f"{ham_probability * 100:.2f}%"
            )

        # Progress bar
        st.progress(float(spam_probability))

st.divider()

st.caption(
    "Built with Python, Scikit-learn, TF-IDF and Multinomial Naive Bayes"
)