import re
import pandas as pd
import numpy as np
from pythainlp.corpus.common import thai_words
from pythainlp import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class SentimentAnalyzer:
    def __init__(self):
        # Load data and models
        self.pos_df = pd.read_csv('model/pos_words.txt', header=None, names=['text'])
        self.neg_df = pd.read_csv('model/neg_words.txt', header=None, names=['text'])
        self.bad_df = pd.read_csv('model/bad_words.txt', header=None, names=['text'])

        self.toxic_words = self.neg_df.values.flatten()
        self.positive_word = self.pos_df.values.flatten()
        self.bad_words = self.bad_df.values.flatten()

        self.words = set(thai_words())
        self.words.update(self.toxic_words)
        self.words.update(self.positive_word)
        self.words.add('ไม่ดี')

        self.stop_words = ['ครับ', 'ค่ะ', 'คะ', 'ครับผม', 'เจ้าค่ะ', 'จ้ะ', 'จ้า']
        self.custom_tokenizer = Tokenizer(self.words)

        # Load models
        self.enc_sentiment = joblib.load('model/transform.pkl')
        self.sentiment_model = joblib.load('model/model.pkl')
        self.label_enc_sentiment = joblib.load('model/label_encoded.pkl')
        self.optimal_threshold_sentiment = np.load('model/sentiment_optimal_thresholds.npy')

    def clean_text(self, text):
        text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r':[a-z_]+:', '', text)
        text = text.strip()
        return text

    def tokenize(self, text):
        final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
        final = self.custom_tokenizer.word_tokenize(final)
        final = " ".join(word for word in final)
        final = " ".join(word for word in final.split() if word.lower() not in self.stop_words)
        return final

    def map_label(self, y):
        if y == 'pos':
            return 'positive'
        elif y == 'neu':
            return 'neutral'
        else:
            return 'negative'

    def check_bad_words(self, input_text):
        for bad_word in self.bad_words:
            if bad_word in input_text.split(' '):
                print('------------------------------------')
                return True
        return False

    def sentiment_classify(self, input_text):
        input_text = self.tokenize(self.clean_text(input_text))

        if self.check_bad_words(input_text):
            return 'negative', 1.0, 'triggered'

        # Transform the input text into TF-IDF representation
        input_tfidf = self.enc_sentiment.transform([input_text])

        # Predict sentiment label and probability
        y_pred = self.sentiment_model.predict(input_tfidf)
        probs = self.sentiment_model._predict_proba_lr(input_tfidf)
        prob = max(probs[0])
        index = np.argmax(probs[0])

        # Convert predicted label index to the original class label
        predicted_label = self.label_enc_sentiment.inverse_transform([y_pred])[0]
        predicted_label = self.map_label(predicted_label)
        
        if prob > self.optimal_threshold_sentiment[index]:
            return predicted_label, prob, 'not triggered'
        else:
            return 'neutral', prob, 'not triggered'

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    text = 'ดูแลดีประดับใจ พนักงานพูดเพราะน่าฟัง แก้ปัญหาได้ดี'
    print("Input text:", text)
    sentiment_pred, sentiment_conf, trigger = analyzer.sentiment_classify(text)
    print("Predicted sentiment:", sentiment_pred)
    print("Confidence:", sentiment_conf)
    print("Trigger:", trigger)
