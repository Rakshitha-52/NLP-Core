import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def load_data():
    # Load dataset
    df = pd.read_csv("spam.csv", encoding='latin1')

    # Keep only needed columns
    df = df[['v1', 'v2']]

    # Convert labels: spam = 1, ham = 0
    df['spam'] = df['v1'].apply(lambda x: 1 if x == 'spam' else 0)

    return df


def train_basic_model(df):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['v2'], df['spam'], test_size=0.2
    )

    # Convert text to vectors
    vectorizer = CountVectorizer()
    X_train_cv = vectorizer.fit_transform(X_train)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_cv, y_train)

    # Test
    X_test_cv = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_cv)

    print("\n=== Basic Model Report ===")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


def test_custom_emails(model, vectorizer):
    emails = [
        "Hello, can you come before 6pm tomorrow?",
        "Upto 50% discount, exclusive offer just for you!"
    ]

    emails_count = vectorizer.transform(emails)
    predictions = model.predict(emails_count)

    print("\n=== Custom Email Predictions ===")
    for email, pred in zip(emails, predictions):
        print(f"Email: {email}")
        print("Prediction:", "Spam" if pred == 1 else "Not Spam")
        print()


def main():
    df = load_data()

    print("Dataset Loaded Successfully!")
    print(df.head())

    model, vectorizer = train_basic_model(df)

    test_custom_emails(model, vectorizer)


if __name__ == "__main__":
    main()

