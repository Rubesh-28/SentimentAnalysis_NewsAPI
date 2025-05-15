import streamlit as st
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import requests
import nltk
from tensorflow.keras.initializers import Orthogonal
import os

# Set the NLTK data path explicitly
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data (moved outside function)
def download_nltk_resources():
    try:
        # Check if punkt is already downloaded
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.write("Downloading punkt...")
        nltk.download('punkt', download_dir=nltk_data_path)  # Specify download directory
        st.write("punkt downloaded successfully.")

    try:
        # Check if stopwords is already downloaded
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.write("Downloading stopwords...")
        nltk.download('stopwords', download_dir=nltk_data_path)  # Specify download directory
        st.write("stopwords downloaded successfully.")

download_nltk_resources()  # Call the function to download resources

# Load the model with custom objects
@st.cache_resource
def load_sentiment_model():
    custom_objects = {'KerasLayer': hub.KerasLayer, 'Orthogonal': Orthogonal}
    try:
        model = load_model('Sentiment.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_sentiment_model()

# Initialize the tokenizer
@st.cache_resource
def create_tokenizer():
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(['your sample text'])  # Important: Fit on *some* text.
    return tokenizer

tokenizer = create_tokenizer()


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
    try:
        response = requests.get(news_api_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        news_data = response.json()
        titles = [article['title'] for article in news_data.get('articles', [])]
        return titles
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []
    except (KeyError, ValueError) as e:  # Catch JSON errors and key errors
        st.error(f"Error processing news data: {e}")
        return []

# Streamlit app
def main():
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
        if model is None:
            st.error("Sentiment model not loaded. Please check for errors.")
            return

        cleaned_text = preprocess_text(text_input)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        max_sequence_length = 100
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

        try:
            sentiment = model.predict(padded_sequence)
            predicted_class = int(sentiment.argmax(axis=-1)[0])

            sentiment_label = "Neutral"
            if predicted_class == 0:
                sentiment_label = "Negative"
            elif predicted_class == 1:
                sentiment_label = "Positive"

            st.subheader("Predicted Sentiment")
            st.write(f"Sentiment: {sentiment_label}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.write("---")
    st.write("Sentiment Analysis by Rubesh K K")

if __name__ == "__main__":
    main()
