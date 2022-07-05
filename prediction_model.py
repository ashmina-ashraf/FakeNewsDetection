from nltk.corpus import stopwords
import joblib
import nltk
import re
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



ps = PorterStemmer()
nltk.data.path.append('./nltk_data')


model = load_model(r"C:\Users\91884\Downloads\Fake-News-Detector-App-master\Fake-News-Detector-App-master\my_model.h5")
print('=> Pickle Loaded : Model ')
#tfidfvect = joblib.load(r"C:\Users\hp\Downloads\Fake-News-Detector-App-master\Fake-News-Detector-App-master\tfidfvect.pkl")
#print('=> Pickle Loaded : Vectorizer')

tokenizer = joblib.load(r"C:\Users\91884\Downloads\Fake-News-Detector-App-master\Fake-News-Detector-App-master\tokenizer.pkl")


class PredictionModel:
    output = {}

    # constructor
    def __init__(self, original_text):
        self.output['original'] = original_text


    # predict
    def predict(self):
        review = self.preprocess()
        #text_vect = tfidfvect.transform([review]).toarray()

        tokenized_text = tokenizer.texts_to_sequences(review)
        padded_text = pad_sequences(tokenized_text,maxlen = 500, padding = 'post', truncating = 'post')
        

        self.output['prediction'] = 'FAKE' if model.predict(np.array(padded_text)) <= 0.5 else 'REAL'
        return self.output


    # Helper methods
    def preprocess(self):
        review = re.sub('[^a-zA-Z]', ' ', self.output['original'])
        review = review.lower()
        review = review.split()
        nltk.download('stopwords')
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        self.output['preprocessed'] = review
        return review
