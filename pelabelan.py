import pandas as pd
from sqlalchemy import create_engine
from pymysql import connect

# Membuat engine koneksi ke MySQL menggunakan SQLAlchemy
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

conn = connect(host='localhost',
               user='root',
               passwd='',
               database='ta2')

# Membuat engine koneksi ke MySQL menggunakan SQLAlchemy
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

# define cursor object
cur = conn.cursor()
# define query to select text from table ikn
query = "SELECT full_text FROM ikn"
# execute query
cur.execute(query)
df = pd.read_sql(query, conn)
df.columns = ["text"]

# Membaca kamus positif dan negatif
positive_lexicon = set(pd.read_csv("kamus/positive.tsv", sep="\t", header=None)[0])
negative_lexicon = set(pd.read_csv("kamus/negative.tsv", sep="\t", header=None)[0])

# Fungsi untuk menentukan sentimen tanpa menghitung skor
def determine_sentiment_and_score(text):
    positive_count = sum(1 for word in text.split() if word in positive_lexicon)
    negative_count = sum(1 for word in text.split() if word in negative_lexicon)
    total_score = positive_count - negative_count
    if total_score > 0:
        sentiment = "Positif"
    elif total_score < 0:
        sentiment = "Negatif"
    else:
        sentiment = "Netral"
    return sentiment, total_score

# Menambahkan kolom sentimen dan skor ke dataframe
df[['label', 'skor']] = df['text'].apply(lambda text: pd.Series(determine_sentiment_and_score(text)))

# Menyimpan ke file CSV dan SQL database
df.to_csv("ikn_labelled.csv", index=False)
df.to_sql(name='labelled', con=engine, if_exists='replace', index=False)