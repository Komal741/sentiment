import streamlit as st
import pickle

# Load pre-trained model and vectorizer
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model and vectorizer files not found! Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop()

# Streamlit UI
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review, and the app will classify it as Positive or Negative.")

# User Input
user_input = st.text_area("Enter your movie review:")

# Analyze Sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Vectorize input and predict sentiment
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a valid review!")
