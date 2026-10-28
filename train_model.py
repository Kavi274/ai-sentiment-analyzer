import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv", encoding="utf-8-sig")

X = data["text"]
y = data["label"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create NLP pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

# Train final model using all available data
model.fit(X, y)

# Save model
joblib.dump(model, "sentiment_model.pkl")

print("Three-class sentiment model trained successfully!")

# Test examples
examples = [
    "Hi",
    "I love this product",
    "This is terrible"
]

for text in examples:
    prediction = model.predict([text])[0]
    print(f"{text} --> {prediction}")
