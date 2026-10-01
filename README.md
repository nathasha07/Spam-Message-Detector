# 📩 Spam Message Detector

A machine learning web application that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing and Multinomial Naive Bayes.

## 🚀 Features

- Classifies SMS messages as spam or ham
- Uses TF-IDF for text feature extraction
- Uses Multinomial Naive Bayes for classification
- Displays spam and ham probabilities
- Includes an interactive Streamlit interface
- Saves the trained model and vectorizer using Joblib

## 🧠 Machine Learning Pipeline

```text
SMS Message
     ↓
Text Data
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Spam / Ham Prediction
     ↓
Probability Display
```

## 📊 Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

The dataset contains 5,572 SMS messages labeled as either:

- **ham** — legitimate message
- **spam** — unwanted message

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Streamlit
- Joblib

## 📈 Model Performance

The model achieved approximately **94.89% accuracy** on the held-out test set.

The dataset was split into:

- **80% training data**
- **20% testing data**

## 💻 Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd spam-message-detector
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
spam-message-detector/
├── data/
│   └── SMSSpamCollection
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔮 Future Improvements

- Improve recall for spam messages
- Add text preprocessing and normalization
- Compare multiple classification algorithms
- Add message history
- Deploy the application online
- Add multilingual SMS classification

## 👩‍💻 Author

Nathasha Vipin
