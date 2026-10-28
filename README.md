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
ai-sentiment-analyzer/
│
├── app.py
├── train_model.py
├── test_model.py
├── dataset.csv
├── sentiment_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

### File Descriptions

| File | Description |
|---|---|
| `app.py` | Streamlit sentiment analysis application |
| `train_model.py` | Trains and saves the machine learning pipeline |
| `test_model.py` | Automated tests for sentiment predictions |
| `dataset.csv` | Labeled sentiment training examples |
| `sentiment_model.pkl` | Serialized trained model |
| `requirements.txt` | Python dependencies |

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kavi274/ai-sentiment-analyzer.git
cd ai-sentiment-analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train_model.py
```

This creates or updates the trained model file.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

Streamlit will display a local URL where you can access the application in your browser.

## 🧪 Automated Testing

The project includes four automated unit tests:

1. `test_three_classes` — Checks that the model supports three sentiment categories.
2. `test_prediction_count` — Checks that predictions match the number of input texts.
3. `test_probability_sum` — Verifies that predicted class probabilities sum to approximately 1.
4. `test_valid_prediction` — Ensures that predicted labels belong to the expected sentiment classes.

Run the tests using:

```bash
python -m unittest test_model -v
```

All four tests passed during local development.

## 📈 Limitations

This is an educational NLP project built using a small, manually labeled dataset. The model may struggle with sarcasm, ambiguous sentences, unfamiliar vocabulary, and complex expressions.

Prediction probabilities should not be interpreted as calibrated measures of real-world certainty.

The project is intended to demonstrate foundational NLP and machine learning concepts rather than production-grade sentiment analysis.

## 🔮 Future Improvements

- Expand the dataset with more diverse real-world text examples
- Evaluate the model using accuracy, precision, recall, and F1-score
- Add a confusion matrix and classification report
- Experiment with advanced NLP models such as BERT
- Improve the Streamlit user interface
- Deploy the application online
- Add automated testing through GitHub Actions

## 👩‍💻 Project Author

**Kaviya — AI & Data Science Undergraduate | AI Engineering Intern**

GitHub: [@Kavi274](https://github.com/Kavi274)

## ⭐ Support

If you find this educational project useful, consider giving the repository a star!

---

**Built with Python, Machine Learning, and a passion for Natural Language Processing.**
