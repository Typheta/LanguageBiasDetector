# 🧠 Language Bias Detector

A machine learning project that detects biased, offensive, or hateful language in user-submitted text. It can also suggest more neutral word replacements. Built using Python, NLP (nltk), Scikit-learn, and a curated hate speech dataset.

---

## 🚀 Features

- Detects biased or offensive language in text
- Provides suggested non-offensive replacements
- Trained on a curated hate speech dataset
- Includes command-line interface for testing
- (Optional) Extendable to a web API using FastAPI

---

## 📁 Project Structure

LanguageBiasDetector/ │ ├── data/ # Dataset goes here │ └── HateSpeechDatasetBalanced.csv │ ├── models/ # Trained model and vectorizer │ ├── bias_detector_model.pkl │ └── vectorizer.pkl │ ├── src/ # Source code modules │ ├── init.py │ ├── preprocess.py │ ├── bias_suggester.py │ ├── train_model.py │ └── detect_api.py # (Optional) FastAPI interface │ ├── venv/ # Virtual environment (not included in repo) │ ├── main.py # Run this to test predictions ├── requirements.txt # All required dependencies └── README.md # You are here


---

## 🧑‍💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LanguageBiasDetector.git
cd LanguageBiasDetector
2. Create a Virtual Environment
python -m venv venv

3. Activate the Environment
Windows (CMD):
venv\Scripts\activate

powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate

macOS/Linux:
source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt
📊 Training the Model
1. Download the dataset from Kaggle:
🔗 Hate Speech Detection - Curated Dataset

Choose HateSpeechDatasetBalanced.csv and place it in the data/ folder.

2. Train and save the model:
python -m src.train_model
✅ Testing with Input
Run the main script to enter a comment and get a prediction:


python main.py
It will output:

Whether the comment is biased

Suggested alternative words (if applicable)

🛠 Dependencies
All dependencies are in requirements.txt. Key packages include:

pandas

numpy

scikit-learn

nltk

joblib

🌐 (Optional) Run as a Web API
Want to serve this as a web service? You can extend it using FastAPI.


uvicorn src.detect_api:app --reload
Then open http://127.0.0.1:8000/docs to test the API.

🧠 Credits
Developed by Jose Espino, Austin Caddell, Braeden McGarvey

Dataset by waalbannyantudre on Kaggle

📄 License
MIT License – Use freely with credit.




