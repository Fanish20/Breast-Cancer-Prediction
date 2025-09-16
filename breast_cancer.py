import os
import argparse
import pickle
import logging

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LogisticRegression on the breast cancer dataset and save the model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use as test set (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="Models/model.pkl", help="Output path for the saved model")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for LogisticRegression")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    return X, y

def build_model(C=1.0, random_state=42):
    # pipeline with scaling + logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state))
    ])
    return pipe

def save_model(obj, out_path):
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    logging.info(f"Model saved to {out_path}")

def main():
    args = parse_args()
    setup_logging()
    logging.info("Loading data...")
    X, y = load_data()

    logging.info(f"Splitting data (test_size={args.test_size}, random_state={args.random_state})")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    logging.info("Building model...")
    model = build_model(C=args.C, random_state=args.random_state)

    logging.info("Training model...")
    model.fit(X_train, y_train)

    logging.info("Evaluating on test set...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info(f"Test accuracy: {acc:.4f}")
    logging.info("Classification report:\n" + classification_report(y_test, preds))

    save_model(model, args.out)
    logging.info("Done.")

if __name__ == "__main__":
    main()
