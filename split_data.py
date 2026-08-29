import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

# Membuat engine koneksi ke MySQL menggunakan SQLAlchemy
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

query = "SELECT stemming, label FROM preprocessing"
try:
    df = pd.read_sql(query, con=engine)
except Exception as e:
    print("Gagal membaca tabel preprocessing:", e)
    df = pd.DataFrame(columns=['stemming', 'label'])

if not df.empty and len(df) >= 5:
    df['stemming'] = df['stemming'].fillna('').astype(str)
    df['label'] = df['label'].fillna('Netral').astype(str)

    # Periksa apakah stratify dimungkinkan (minimal 2 sample per class)
    value_counts = df['label'].value_counts()
    can_stratify = (value_counts >= 2).all() and len(value_counts) > 1

    # Membelah data menggunakan train_test_split
    if can_stratify:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["stemming"], df["label"], test_size=0.2, stratify=df["label"], random_state=30
        )
    else:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["stemming"], df["label"], test_size=0.2, random_state=30
        )

    # Membuat dataframe training dan testing
    train_df = pd.DataFrame({'text': X_train_text, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test_text, 'label': y_test})

    # Menyimpan data training dan testing ke SQL database
    train_df.to_sql(name='train', con=engine, if_exists='replace', index=False)
    test_df.to_sql(name='test', con=engine, if_exists='replace', index=False)
    print("Pemisahan data (train/test split) berhasil!")
else:
    print("Data preprocessing kosong atau tidak cukup untuk di-split.")