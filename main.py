from src.preprocess import clean_text
from src.bias_suggester import suggest_replacements
import joblib
import os

# Load trained model and vectorizer
model = joblib.load("models/bias_detector_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load and print model metrics
metrics_path = "models/metrics.txt"
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        print("🔍 Model Evaluation Metrics:")
        print(f.read())
else:
    print("⚠️ No model metrics file found.")

# Optional: Display saved evaluation plot (requires matplotlib)
try:
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open("models/model_evaluation.png")
    plt.imshow(img)
    plt.axis('off')
    plt.title("Model Evaluation Results")
    plt.show()
except Exception as e:
    print("Visualization not available:", e)

# Get input from user
text = input("\nEnter a comment to analyze: ")

# Clean and transform input
cleaned = clean_text(text)
vec = vectorizer.transform([cleaned])

# Predict
probability = model.predict_proba(vec)[0][1]  # probability of being biased
prediction = model.predict(vec)[0]
bias_detected = bool(prediction)

# Output
print("\n🧠 Bias Detected:", bias_detected)
print(f"Confidence: {probability * 100:.2f}%")

if bias_detected:
    print("\n💡 Suggested Replacement:")
    print(suggest_replacements(text))
else:
    print("\n✅ No biased language detected.")
