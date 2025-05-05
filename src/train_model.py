import pandas as pd
import os
import re
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

def train_and_save_model():
    print("📁 Loading dataset...")
    dataset_path = "data/HateSpeechDatasetBalanced.csv"
    df = pd.read_csv(dataset_path)

    # Optional: limit size for faster dev/test
    df = df.sample(10000, random_state=42)

    df = df[df['Label'].isin([0, 1])]
    df = df.rename(columns={"Content": "text", "Label": "label"})
    df['cleaned'] = df['text'].apply(clean_text)

    X = df['cleaned']
    y = df['label']

    print("🧹 Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    print("📈 Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_vec, y)

    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    print("🔍 Running GridSearchCV for Logistic Regression...")
    param_grid = {
        'C': [0.1, 1.0, 10],
        'solver': ['liblinear']
    }
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring='f1_weighted')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("✅ Best model trained. Evaluating...")
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)

    print("💾 Saving model and vectorizer...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/bias_detector_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    with open("models/metrics.txt", "w") as f:
        f.write("Best Logistic Regression Model:\n")
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.4f}\n")

    print("📊 Saving ROC and Confusion Matrix plot...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axs[0], cmap='Blues', values_format='d')
    axs[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    axs[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axs[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].set_title("ROC Curve")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("models/model_evaluation.png")
    print("✅ Plot saved as: models/model_evaluation.png")

if __name__ == "__main__":
    train_and_save_model()

