import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle  # If using a tokenizer from training

# Load trained model
model = tf.keras.models.load_model("fake_review_model_tf.h5")

# Load the tokenizer (Ensure you saved it during training)
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define FastAPI app
app = FastAPI()

# Define input model
class ReviewInput(BaseModel):
    review: str

# Preprocessing function
def preprocess_text(text):
    # Tokenize the text (Modify based on training)
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=512)  # Adjust `maxlen`
    return padded_seq

# Prediction endpoint
@app.post("/predict")
def predict_review(data: ReviewInput):
    try:
        processed_text = preprocess_text(data.review)
        prediction = model.predict(processed_text)
        label = "Fake Review" if prediction[0] > 0.5 else "Real Review"
        return {"review": data.review, "prediction": label}
    except Exception as e:
        return {"error": str(e)}
