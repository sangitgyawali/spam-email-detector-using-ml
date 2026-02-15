import pandas as pd
import string
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)  
data = pd.read_csv("spam.csv", encoding='latin-1')

data['label'] = data['label'].map({'ham':0, 'spam':1})

data['message'] = data['message'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")

print("\n=== Spam Detector Ready ===")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

while True:
    msg = input("\nEnter email text (or type exit): ")

    if msg.lower() == "exit":
        break

    msg_clean = clean_text(msg)
    msg_vector = vectorizer.transform([msg_clean])

    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("🚨 Spam Email")
    else:
        print("✅ Not Spam")