from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import requests,nltk
app = Flask(__name__)
model = load_model('model.h5')

nltk.download('punkt')
nltk.download('stopwords')



@app.route('/fetch_news', methods=['GET'])
def fetch_news():
    keyword = request.args.get('keyword')
    news_api_url = f'https://newsapi.org/v2/everything?q={keyword}&apiKey=YOUR_NEWS_API_KEY'
    response = requests.get(news_api_url)
    if response.status_code == 200:
        news_data = response.json()
        titles = [article['title'] for article in news_data.get('articles', [])]
        return jsonify(titles=titles)
    else:
        return jsonify(error="Failed to fetch news titles")

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    words = word_tokenize(text.lower())
    filtered_words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words]
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

max_sequence_length = 100 

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)
    text = data['text']

    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    sentiment = model.predict(padded_sequence)
    predicted_class = int(sentiment.argmax(axis=-1)[0])
    return jsonify(sentiment=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
