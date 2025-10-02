# 📰 Fake News Detection using Neural Networks

This project builds a binary text classifier to detect whether a given news article is **real or fake**.  
It combines **Natural Language Processing (NLP)** techniques with a **Neural Network (ANN)** built using Keras.

---

## 🔧 Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- NLTK
- CountVectorizer / TF-IDF
- Streamlit (for interactive UI)
- Newspaper3k (for scraping articles from URLs)

---

## 📊 Dataset
The dataset used is from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), which contains labeled news articles (`True` or `Fake`) from multiple sources.  

- **True news** → assigned label `1`  
- **Fake news** → assigned label `0`

---

## 🧠 Model Overview
- **Input:** Preprocessed news text (title + content combined)  
- **Preprocessing:** Lowercasing, stopword removal, stemming, tokenization  
- **Vectorization:** TF-IDF / CountVectorizer  
- **Model:** Dense Artificial Neural Network with:
  - ReLU activations  
  - Dropout layers for regularization  
  - Final sigmoid layer for binary output  
- **Optimizer:** Adam  
- **Loss:** Binary Crossentropy  

---

## 🎯 Accuracy
- Training Accuracy: ~99%  
- Validation Accuracy: ~98–99% (on Kaggle dataset)  

(*Note: Real-world performance may vary, especially with satire or unseen sources like **The Onion**.*)

---

## ⚡ Features
- ✅ Data preprocessing (cleaning + stemming + tokenization)  
- ✅ Fake vs. Real classification  
- ✅ Model + vectorizer are saved and reused (`.keras` + `.pkl`)  
- ✅ Detects news by **URL** (scraped via Newspaper3k) or **manual text input**  
- ✅ Streamlit-powered web UI for easy testing  

---

## 💻 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/fake-news-classifier.git
cd fake-news-classifier
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the model (if not already trained)

python train.py
    model/Fake_news_mod.keras (trained model)

    model/vectorizer.pkl (vectorizer)

### 4. Run Streamlit app
```bash
streamlit run app.py