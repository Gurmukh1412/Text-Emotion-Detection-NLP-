# 🧠 Text Emotion Detection using Machine Learning

A complete end-to-end NLP project that detects emotions from text using Machine Learning.

This project classifies user input text into emotions such as:

- 😄 Joy  
- 😢 Sadness  
- 😡 Anger  
- 😨 Fear  
- 😍 Love  
- 😲 Surprise

---

## 🚀 Live Demo

🔗 Deployed App Link: **https://text-emotion-detection-nlp-00.streamlit.app/**

---

## 📌 Project Overview

Text Emotion Detection is a Natural Language Processing (NLP) application that analyzes textual input and predicts the underlying emotion.

The project includes:

- Data preprocessing
- Text cleaning
- Tokenization
- Feature extraction (TF-IDF)
- Model training
- Model serialization
- Web app deployment (if applicable)

---

## 🛠️ Tech Stack

- Python 🐍
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Flask / Streamlit (if used)
- Pickle (for model saving)

---

## 📂 Project Structure

```
Text-Emotion-Detection/
│
├── emotion_dataset_raw.csv        # Raw dataset
├── text_emotion.pkl               # Trained ML model
├── Text Emotion Detection.ipynb   # Training Notebook
├── app.py                         # Web app (if applicable)
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 📊 Dataset

The dataset contains text samples labeled with emotions.

Example:

| Text | Emotion |
|------|---------|
| I feel amazing today | Joy |
| I am very disappointed | Sadness |
| This makes me so angry | Anger |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Gurmukh1412/text-emotion-detection.git
cd text-emotion-detection
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

If using Flask:

```bash
python app.py
```

If using Streamlit:

```bash
streamlit run app.py
```

---

## 🧠 Model Training Pipeline

1. Text Cleaning (lowercase, remove punctuation, stopwords)
2. Tokenization
3. TF-IDF Vectorization
4. Train-Test Split
5. Model Training (Logistic Regression / Naive Bayes / etc.)
6. Model Evaluation
7. Model Saving using Pickle

---

## 📈 Model Performance

- Accuracy: Add your accuracy here
- Evaluation Metrics:
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

---

## 🔥 Features

✔ Clean and simple UI  
✔ Real-time emotion prediction  
✔ Pre-trained model  
✔ Easy deployment  
✔ Scalable architecture  

---

## 📌 Example Usage

Input:
```
I am feeling very happy today!
```

Output:
```
Predicted Emotion: Joy 😄
```

---

## 💡 Future Improvements

- Deep Learning (LSTM / BERT)
- Multi-label emotion classification
- Emotion intensity detection
- API deployment
- Docker containerization

---

## 👨‍💻 Author

**Gurmukh Singh**  
GitHub: https://github.com/Gurmukh1412  

---

## ⭐ If you found this project useful

Give it a ⭐ on GitHub!
