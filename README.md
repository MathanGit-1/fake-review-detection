# 🕵️‍♂️ Fake Review Detection using Deep Learning & NLP

## 📌 Project Overview
This project detects **fake reviews** using **Deep Learning (TensorFlow)** and **Natural Language Processing (NLP)** techniques.  
It leverages the **Universal Sentence Encoder (USE)** from TensorFlow Hub for embeddings and a **Neural Network (ANN)** for classification.  

A **Streamlit Web App** is also built for easy user interaction, allowing users to input reviews and check if they are **fake or real**.

---

## 🔥 Technologies Used
- 🧠 **Deep Learning:** TensorFlow, Keras
- 🔡 **NLP Processing:** NLTK, TensorFlow Hub (Universal Sentence Encoder)
- 📊 **Data Handling:** Pandas, NumPy
- 🖥 **Model Deployment:** Streamlit, FastAPI
- 🌍 **Web API Calls:** Requests

---

## 📂 Folder Structure
```bash
FakeReviewDetection/
│── model_training.ipynb        # Jupyter Notebook for model training
│── data_preprocessing.py        # Data cleaning & text preprocessing
│── app.py                       # Streamlit UI for user input
│── fake_review_model_tf.h5       # Trained deep learning model
│── requirements.txt              # Dependencies
│── README.md                     # Project documentation
│── fake_reviews_dataset.csv      # Dataset (if included)
---

## 🛠 Installation & Setup
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/MathanGit-1/fake-review-detection.git
cd fake-review-detection

##2️⃣ Create a Virtual Environment
python -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate     # For Windows

#Install Dependencies
pip install -r requirements.txt

# Train the Model
jupyter notebook
# Open model_training.ipynb and run all cells

#Start the Streamlit Web App
streamlit run app.py

#API Usage (If Using FastAPI)
Endpoint	Method	Description
/predict	POST	Predict if a review is fake or real

#Exampe API call
import requests
response = requests.post("http://localhost:8000/predict", json={"review": "This product is amazing!"})
print(response.json())


#Future Improvements
✅ Improve model performance with transformer-based models (BERT, GPT-3)
✅ Deploy the model on Hugging Face Spaces / AWS / GCP

✨ Author
👤 Mathan
📧 Contact: mathankumar137@gmail.com
🔗 GitHub: MathanGit-1
🔗 LinkedIn: www.linkedin.com/in/mathankumar-r-9bb5a118a