import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Product Review Sentiment Analysis")
st.write("Enter a product review to predict its sentiment")

review = st.text_area("Review Text")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = preprocess(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")
if __name__=="__main__":
    pass