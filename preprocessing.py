from pymysql import connect
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from sqlalchemy import create_engine
import logging

# connect to the database
conn = connect(host='localhost',
               user='root',
               passwd='',
               database='ta2')

# Membuat engine koneksi ke MySQL menggunakan SQLAlchemy
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

# define cursor object
cur = conn.cursor()
# define query to select text from table labelled
query = "SELECT * FROM labelled"
# execute query
cur.execute(query)
df = pd.read_sql(query, conn)

# Handle NaN values in the 'text' column
df['text'] = df['text'].fillna('')

# casefolding
df['text_clean'] = df['text'].str.lower()

# Cleansing
def remove_tweet_special(text):
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', " ")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\\w+:\\/\\/\\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

df['text_clean'] = df['text_clean'].apply(remove_tweet_special)

# Menghilangkan data duplikat
df.drop_duplicates(subset='text_clean', keep='first', inplace=True)

# Remove number
df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r"\d", "", x))

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
nltk.download('stopwords')
list_stopwords = set(stopwords.words('indonesian'))
# Menambahkan stopwords tambahan
list_stopwords.update(["prabowogibran", "prabowo", "anies", "ganjar", "lohh", "loh", "ahhh", "aaah", "ae", "yang", "nih", "ah", "wkwkwk",
                       "wkwk", "wk", "wkwkwkwk", "lhoo", "lho", "ah", "ahh", "lohh", "ahm", "sih", "ya", "eh"])
print(list_stopwords)
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
df['stemming'] = df['stopwords'].swifter.apply(lambda x: [stemmed_wrapper(term) for term in x])

# Convert list of tokens to string with spaces
df['token'] = df['token'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)
df['normalisasi'] = df['normalisasi'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)
df['stopwords'] = df['stopwords'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)
df['stemming'] = df['stemming'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)

# Memindahkan kolom label ke paling kanan
columns = [col for col in df.columns if col != 'label'] + ['label']
df = df[columns]

# Menyimpan ke dalam CSV dan SQL
df.to_csv("ikn_prepro.csv", index=False)
df.to_sql(name='preprocessing', con=engine, if_exists='replace', index=False)

# # Menggabungkan teks asli dengan label untuk analisis lebih lanjut
# df_text = pd.read_csv('ikn_prepro.csv', usecols=['text'])
# df_label = pd.read_csv('ikn_labelled.csv', usecols=['label'])
# df_merged = pd.concat([df_text, df_label], axis=1)
# df_merged.columns = ["text", "label"]

# # Menyimpan hasil gabungan ke file Excel
# df_merged.to_excel('data_sentimen_ikn.xlsx', index=False)

# # Fungsi untuk menghitung jumlah data per label
# def count_labels(df):
#     label_counts = df['label'].value_counts()
#     print("Jumlah data per label:")
#     print(label_counts)
#     return label_counts
# label_counts = count_labels(df)

# df_label_counts = pd.DataFrame(label_counts).reset_index()
# df_label_counts.columns = ['Label', 'Jumlah']
# df_label_counts.to_csv("label_counts.csv", index=False)

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
