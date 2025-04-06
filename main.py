from src.preprocess import clean_text
from src.bias_suggester import suggest_replacements
import joblib

# Load trained model and vectorizer
model = joblib.load("models/bias_detector_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Get input from user
text = input("Enter a comment: ")

# Clean and transform
cleaned = clean_text(text)
vec = vectorizer.transform([cleaned])

# Predict
prediction = model.predict(vec)
bias_detected = bool(prediction[0])

# Output
print("\nBias Detected:", bias_detected)

if bias_detected:
    print("Suggested Replacement:\n", suggest_replacements(text))
