import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from sqlalchemy import create_engine
import logging

# Ensure NLTK resources are available
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Membuat engine koneksi ke MySQL menggunakan SQLAlchemy
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

# Define query to select text and label from table labelled
query = "SELECT text, label FROM labelled"
try:
    df = pd.read_sql(query, con=engine)
except Exception as e:
    print("Gagal membaca tabel labelled:", e)
    df = pd.DataFrame(columns=['text', 'label'])

if not df.empty:
    # Handle NaN values in the 'text' and 'label' columns
    df['text'] = df['text'].fillna('').astype(str)
    df['label'] = df['label'].fillna('Netral').astype(str)

    # casefolding
    df['text_clean'] = df['text'].str.lower()

    # Cleansing
    def remove_tweet_special(text):
        text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', " ")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
        return text.replace("http://", " ").replace("https://", " ")

    df['text_clean'] = df['text_clean'].apply(remove_tweet_special)

    # Menghilangkan data duplikat berdasarkan text_clean
    df.drop_duplicates(subset='text_clean', keep='first', inplace=True)

    # Remove number
    df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r"\d+", "", x))

    # Remove punctuation
    df['text_clean'] = df['text_clean'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

    # Remove whitespace leading & trailing
    df['text_clean'] = df['text_clean'].apply(lambda x: x.strip())

    # Remove multiple whitespace into single whitespace
    df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # NLTK word tokenize
    df['token'] = df['text_clean'].apply(word_tokenize)

    # NORMALISASI
    kamus_normalisasi = pd.read_csv("kamus/slang.csv")
    kata_normalisasi_dict = {row[0]: row[1] for index, row in kamus_normalisasi.iterrows()}

    def normalisasi_kata(document):
        return [kata_normalisasi_dict.get(term, term) for term in document]

    df['normalisasi'] = df['token'].apply(normalisasi_kata)

    # Stopwords
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(["prabowogibran", "prabowo", "anies", "ganjar", "lohh", "loh", "ahhh", "aaah", "ae", "yang", "nih", "ah", "wkwkwk",
                           "wkwk", "wk", "wkwkwkwk", "lhoo", "lho", "ah", "ahh", "lohh", "ahm", "sih", "ya", "eh", "yg", "dgn", "bgt"])

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    df['stopwords'] = df['normalisasi'].apply(stopwords_removal)

    # Create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    # Apply stemming
    df['stemming'] = df['stopwords'].swifter.apply(lambda x: [stemmed_wrapper(term) for term in x if term])

    # Convert list of tokens to string with spaces
    df['token'] = df['token'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
    df['normalisasi'] = df['normalisasi'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
    df['stopwords'] = df['stopwords'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
    df['stemming'] = df['stemming'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))

    # Pastikan urutan kolom sesuai dengan tabel dan template preprocessing.html:
    # 0: text, 1: text_clean, 2: token, 3: normalisasi, 4: stopwords, 5: stemming, 6: label
    columns = ['text', 'text_clean', 'token', 'normalisasi', 'stopwords', 'stemming', 'label']
    df = df[columns]

    # Menyimpan ke dalam CSV dan SQL
    df.to_csv("ikn_prepro.csv", index=False)
    df.to_sql(name='preprocessing', con=engine, if_exists='replace', index=False)
    print("Preprocessing selesai dan data berhasil disimpan!")
else:
    print("Data labelled kosong, preprocessing dilewati.")
