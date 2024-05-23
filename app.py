import streamlit as st
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import requests
import nltk
from tensorflow.keras.initializers import Orthogonal

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the model with custom objects
custom_objects = {'Orthogonal': Orthogonal}
model = load_model('model.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(['your sample text'])

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    words = word_tokenize(text.lower())
    filtered_words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words]
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text

# Function to fetch news
def fetch_news(keyword):
    news_api_url = f'https://newsapi.org/v2/everything?q={keyword}&apiKey=b6c9af69e1bb4f45ba61a003a71b20b0'
    response = requests.get(news_api_url)
    if response.status_code == 200:
        news_data = response.json()
        titles = [article['title'] for article in news_data.get('articles', [])]
        return titles
    else:
        return []

# Streamlit app
st.title("Sentiment Analysis")

# Fetch News Section
st.header("Fetch News")
keyword = st.text_input("Enter keyword:")
if st.button("Fetch News"):
    news_titles = fetch_news(keyword)
    if news_titles:
        st.subheader("News Titles")
        for title in news_titles:
            st.write(title)
    else:
        st.write("Failed to fetch news titles")

# Sentiment Analysis Section
st.header("Predict Sentiment")
text_input = st.text_area("Enter your text:")
if st.button("Predict Sentiment"):
    cleaned_text = preprocess_text(text_input)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    max_sequence_length = 100
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    sentiment = model.predict(padded_sequence)
    predicted_class = int(sentiment.argmax(axis=-1)[0])

    sentiment_label = "Neutral"
    if predicted_class == 0:
        sentiment_label = "Negative"
    elif predicted_class == 1:
        sentiment_label = "Positive"

    st.subheader("Predicted Sentiment")
    st.write(f"Sentiment: {sentiment_label}")

st.write("---")
st.write("Sentiment Analysis by Rubesh K K")
