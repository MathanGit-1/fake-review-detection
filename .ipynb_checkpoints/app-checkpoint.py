from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("fake_review_model_tf.h5")

app = FastAPI()

class ReviewInput(BaseModel):
    review: str

@app.post("/predict")
def predict_review(data: ReviewInput):
    # Preprocess input (convert text to numerical data)
    processed_input = np.array([[len(data.review)]])  # Replace with actual NLP preprocessing
    prediction = model.predict(processed_input)
    return {"prediction": float(prediction[0][0])}
