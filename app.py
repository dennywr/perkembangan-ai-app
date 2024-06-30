import streamlit as st
import emoji
import pickle
import regex
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('best_svm_model.pkl', 'rb') as f:
    best_svm_model = pickle.load(f)

# preprocessing
def add_space_around_emoji(text):
    text = regex.sub(r'(\S)(\p{Emoji_Presentation})', r'\1 \2', text)
    text = regex.sub(r'(\p{Emoji_Presentation})(\S)', r'\1 \2', text)
    text = regex.sub(r'\s+', ' ', text)
    return text

def emoji2description(text):
    text = add_space_around_emoji(text)
    return emoji.replace_emoji(text, replace=lambda chars, data_dict: ' '.join(data_dict['id'].split('_')).strip(':'))

def casefolding(text):
    text = text.lower()
    return text

def cleaning(text):
    text = text.replace('\\t', "").replace('\\n', "").replace('\\u', "").replace('\\', "")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = re.sub(r'https?://\S+|www\.\S+|t\.co/\S+', '', text)
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    text = re.sub(r"\d+", "", text)
    return text

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

url = 'https://raw.githubusercontent.com/dennywr/data/main/Normalization%20Data.csv'
slank_words_df = pd.read_csv(url, sep=';')
slank_words_dict = dict(zip(slank_words_df['Slangword'], slank_words_df['Kata Baku']))

def normalize(tokens):
    normalized_tokens = [slank_words_dict.get(token, token) for token in tokens]
    return normalized_tokens

stopwords_indonesia = set(stopwords.words('indonesian'))

def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords_indonesia]
    return ' '.join(filtered_tokens)

def stem(text):
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    return stemmer.stem(text)

st.subheader("Analisis Sentimen Perkembangan AI")
st.write("Oleh | Denny Wahyudi Ramadhon | 200411100170")

new_data = st.text_input("Masukkan teks:")

if st.button("Prediksi"):
    if new_data:
        new_data_no_emoji = emoji2description(new_data)
        new_data_cleaned = cleaning(new_data_no_emoji)
        new_data_casefolded = casefolding(new_data_cleaned)
        tokens = tokenize(new_data_casefolded)
        normalized_tokens = normalize(tokens)
        filtered_text = remove_stopwords(normalized_tokens)
        stemmed_text = stem(filtered_text)

        st.write(f"Teks hasil preprocessing: {stemmed_text}")

        new_data_transformed = tfidf_vectorizer.transform([stemmed_text])
        predicted_labels = best_svm_model.predict(new_data_transformed)

        st.write(f"Label hasil prediksi: {predicted_labels[0]}")
    else:
        st.write("Tolong masukkan teks terlebih dahulu.")
