
import unittest
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TestSentimentModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_path = Path(__file__).parent / "dataset.csv"
        data = pd.read_csv(dataset_path, encoding="utf-8-sig")

        cls.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

        cls.model.fit(data["text"], data["label"])

    def test_three_classes(self):
        self.assertEqual(
            set(self.model.classes_),
            {"positive", "negative", "neutral"}
        )

    def test_prediction_count(self):
        predictions = self.model.predict([
            "I love this product",
            "This is terrible",
            "Hello"
        ])
        self.assertEqual(len(predictions), 3)

    def test_probability_sum(self):
        probabilities = self.model.predict_proba(
            ["I love this product"]
        )[0]
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_valid_prediction(self):
        prediction = self.model.predict(["Hello"])[0]
        self.assertIn(
            prediction,
            ["positive", "negative", "neutral"]
        )


if __name__ == "__main__":
    unittest.main()
