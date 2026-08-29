import pandas as pd
from sqlalchemy import create_engine
import os
import re
import string

# Membuat engine koneksi ke MySQL menggunakan SQLAlchemy
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

query = "SELECT full_text FROM ikn"
try:
    df = pd.read_sql(query, con=engine)
    df.columns = ["text"]
except Exception as e:
    print("Gagal membaca tabel ikn:", e)
    df = pd.DataFrame(columns=["text"])

if not df.empty:
    # Filter baris header jika ada
    df = df[~df['text'].astype(str).str.strip().str.lower().isin(['full_text', 'fulltext', 'tweet', 'text', 'created_at', ''])].copy()
    df['text'] = df['text'].fillna('').astype(str)

    # Membaca kamus positif dan negatif
    pos_file = "kamus/positive.tsv"
    neg_file = "kamus/negative.tsv"
    
    if os.path.exists(pos_file):
        positive_lexicon = set(pd.read_csv(pos_file, sep="\t", header=None)[0].astype(str).str.lower().str.strip())
    else:
        positive_lexicon = set()

    if os.path.exists(neg_file):
        negative_lexicon = set(pd.read_csv(neg_file, sep="\t", header=None)[0].astype(str).str.lower().str.strip())
    else:
        negative_lexicon = set()

    # Fungsi untuk menentukan sentimen dan skor dengan tokenisasi bersih
    def determine_sentiment_and_score(text):
        cleaned = re.sub(r"https?://\S+|@\w+|#\w+|\d+", " ", str(text).lower())
        words = [w.strip(string.punctuation) for w in cleaned.split() if w.strip(string.punctuation)]
        
        positive_count = sum(1 for word in words if word in positive_lexicon)
        negative_count = sum(1 for word in words if word in negative_lexicon)
        total_score = positive_count - negative_count

        if total_score > 0:
            sentiment = "Positif"
        elif total_score < 0:
            sentiment = "Negatif"
        else:
            sentiment = "Netral"
        return sentiment, total_score

    # Menambahkan kolom sentimen dan skor ke dataframe
    results = df['text'].apply(determine_sentiment_and_score)
    df['label'] = [r[0] for r in results]
    df['skor'] = [r[1] for r in results]

    # Menyimpan ke file CSV dan SQL database
    df.to_csv("ikn_labelled.csv", index=False)
    df.to_sql(name='labelled', con=engine, if_exists='replace', index=False)
    print(f"Pelabelan selesai: {len(df)} data berhasil dilabeli!")
    print(df['label'].value_counts())
else:
    print("Tabel ikn kosong, pelabelan dilewati.")