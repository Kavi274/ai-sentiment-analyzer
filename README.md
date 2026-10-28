# 🧠 AI Sentiment Analyzer

### Machine Learning-Based Sentiment Classification with Python & Streamlit

A beginner-friendly Natural Language Processing (NLP) application that analyzes text and predicts whether the sentiment is **Positive 😊**, **Negative 😞**, or **Neutral 😐**.

This project demonstrates a complete machine learning workflow, including text preprocessing, TF-IDF feature extraction, model training, sentiment prediction, and deployment through an interactive Streamlit web interface.

---

## 🚀 Features

- 📝 Classifies user-entered text into **Positive, Negative, or Neutral**
- 🤖 Uses **Logistic Regression** for sentiment classification
- 🔤 Converts text into numerical features using **TF-IDF Vectorization**
- 📊 Provides sentiment prediction probabilities
- 🌐 Interactive web interface built with **Streamlit**
- 💾 Saves the trained machine learning model using **Pickle**
- 🧪 Includes automated unit tests for model validation

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Scikit-learn | Machine learning model and TF-IDF vectorization |
| Pandas | Dataset loading and manipulation |
| Streamlit | Interactive web application |
| Pickle | Model serialization |
| Unittest | Automated model testing |
| Git & GitHub | Version control and collaboration |

## 🧠 How It Works

The application follows a traditional machine learning-based NLP pipeline:

**User Text → TF-IDF Vectorization → Logistic Regression → Sentiment Prediction**

### 1. Dataset Preparation

The model is trained using a small labeled dataset containing three sentiment categories:

- Positive
- Negative
- Neutral

The initial dataset contains 45 labeled text examples, with 15 examples per sentiment class.

### 2. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) transforms text into numerical features based on the importance of words within the dataset.

### 3. Model Training

A Logistic Regression classifier learns relationships between text features and their corresponding sentiment labels.

### 4. Sentiment Prediction

When users enter text into the application, the trained model predicts a sentiment category and produces class probabilities.

### 5. Streamlit Interface

The Streamlit application provides an interactive interface for entering text and viewing the model's predictions.

---

## 📂 Project Structure

```text
