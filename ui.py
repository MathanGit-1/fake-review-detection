import streamlit as st
import requests

# Streamlit UI
st.title("🕵️‍♂️ Fake Review Detection")
st.write("Enter a review below to check if it's real or fake.")

review_text = st.text_area("Enter your review:", "")

if st.button("Check Review"):
    if review_text.strip():
        response = requests.post("http://localhost:8000/predict", json={"review": review_text})
        result = response.json()
        st.write("### Prediction:")
        st.success(result["prediction"])
    else:
        st.warning("Please enter a review.")
