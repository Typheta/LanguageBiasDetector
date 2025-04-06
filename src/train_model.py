import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from src.preprocess import clean_text

def train_and_save_model():
    # load dataset
    dataset_path = "data/HateSpeechDatasetBalanced.csv"
    df = pd.read_csv(dataset_path)

    # print column names and first few rows
    print("Columns:", df.columns)
    print(df.head())

    # rename columns for consistency
    df = df.rename(columns={"Content": "text", "Label": "label"})

    # clean the text
    df['cleaned'] = df['text'].apply(clean_text)

    # features and labels
    X = df['cleaned']
    y = df['label']

    # vectorize text
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    #train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred))

    # save model and vectorizer
    joblib.dump(model, "models/bias_detector_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("\nModel and vectorizer saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
