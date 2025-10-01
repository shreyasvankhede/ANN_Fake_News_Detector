# 📰 Fake News Detection using Neural Networks

This project builds a binary text classifier to detect whether a given news article is real or fake. It uses basic Natural Language Processing techniques and a simple Artificial Neural Network built with Keras.

## 🔧 Technologies Used
- Python
- Keras (TensorFlow backend)
- Scikit-learn
- Pandas, NumPy
- CountVectorizer / TF-IDF
- Gradio (for optional UI)

## 📊 Dataset
The dataset used is from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), which contains labeled news articles from various sources.

## 🧠 Model Overview
- Input: Preprocessed news text (title + content)
- Output: Binary classification — `Real` or `Fake`
- Architecture: Dense ANN with dropout and ReLU activations
- Optimizer: Adam

## 🎯 Accuracy
Achieved ~0% validation accuracy after training (will update once trained).

## ⚡ Features
- Cleaned and preprocessed news text
- Word vectorization using TF-IDF
- ANN classification
- Gradio-powered web UI for real-time demo

## 💻 Run Locally
```bash
git clone https://github.com/yourusername/fake-news-classifier.git
cd fake-news-classifier
pip install -r requirements.txt
python app.py  # or jupyter notebook
