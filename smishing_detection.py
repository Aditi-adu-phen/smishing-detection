"""
SmishGuard - Smishing Detection System
Classifies SMS messages and URLs as spam (smishing) or ham (legitimate)
using TF-IDF features with Multinomial Naive Bayes, SVM and KNN.

Datasets (download separately, place next to this script):
  - sms_spam.csv   : SMS Spam Collection (Kaggle: "SMS Spam Collection Dataset", uciml)
  - links_spam.csv : labeled URL dataset (spam/ham or malicious/benign links)

Edit the CONFIG block below to match the column names of your CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# ----------------------------- CONFIG ---------------------------------
SMS_FILE = "sms_spam.csv"
LINKS_FILE = "links_spam.csv"

# Kaggle's SMS Spam Collection uses columns "v1" (label) and "v2" (text).
# If your file already has "label" and "Text", leave the maps as they are.
SMS_COLUMNS = {"v1": "label", "v2": "Text", "label": "label", "Text": "Text"}
LINKS_COLUMNS = {"url": "Text", "URL": "Text", "text": "Text", "Text": "Text",
                 "type": "label", "Label": "label", "label": "label"}

# Map whatever labels your files use onto "ham" / "spam".
LABEL_MAP = {
    "ham": "ham", "spam": "spam",
    "legitimate": "ham", "benign": "ham", "good": "ham", "0": "ham",
    "smishing": "spam", "phishing": "spam", "malicious": "spam",
    "bad": "spam", "defacement": "spam", "malware": "spam", "1": "spam",
}
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ----------------------------------------------------------------------


def load_dataset(path, column_map):
    """Read a CSV, standardise column names to Text/label, clean labels."""
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns={c: column_map[c] for c in df.columns if c in column_map})
    df = df[["Text", "label"]].dropna(subset=["Text", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    return df.dropna(subset=["label"])


# Step 1: Data collection and preprocessing
df_sms = load_dataset(SMS_FILE, SMS_COLUMNS)
df_links = load_dataset(LINKS_FILE, LINKS_COLUMNS)
df = pd.concat([df_sms, df_links], ignore_index=True)
df.dropna(subset=["Text"], inplace=True)
print(f"Total samples: {len(df)}")
print(df["label"].value_counts(), "\n")

# Step 2: Feature engineering (TF-IDF)
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(df["Text"])
y = df["label"]

# Step 3: Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Step 4: Train and evaluate models
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Support Vector Machine": LinearSVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
}

accuracies = {}
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)
    fitted[name] = model
    print(f"Accuracy ({name}): {accuracies[name]:.4f}")
    print(classification_report(y_test, y_pred))

# Step 5: Real-time prediction on new messages and links
new_messages = [
    "Congratulations! You've won a prize. Click here to claim.",
    "Don't forget our meeting tomorrow at 10 AM.",
    "http://example.com/offer",
    "http://example.com/free-gift",
]
new_tfidf = tfidf_vec.transform(new_messages)
best_name = max(accuracies, key=accuracies.get)
predictions = fitted[best_name].predict(new_tfidf)
print(f"Predictions using {best_name}:")
for message, prediction in zip(new_messages, predictions):
    print(f"Message: {message}\nPrediction: {prediction.capitalize()}\n")

# Step 6: Visualisation (saved so it can be added to the README)
plt.figure(figsize=(7, 4))
plt.bar(accuracies.keys(), accuracies.values(), color=["blue", "green", "orange"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparison of Classifier Accuracies")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()

ConfusionMatrixDisplay.from_estimator(fitted[best_name], X_test, y_test)
plt.title(f"Confusion Matrix ({best_name})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
