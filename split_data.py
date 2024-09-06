import pandas as pd
from pymysql import connect
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

# connect to the database
conn = connect(host='localhost',
               user='root',
               passwd='',
               database='ta2')

# Membuat engine koneksi ke MySQL menggunakan SQLAlchemy
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

cur = conn.cursor()
query = "SELECT stemming, label FROM preprocessing"
cur.execute(query)
df = pd.read_sql(query, conn)

# Membelah data menggunakan stratified sampling
X_train_text, X_test_text, y_train, y_test = train_test_split(df["stemming"], df["label"],
                                                              test_size=0.2, stratify=df["label"], random_state=30)

# Membuat dataframe training dan testing
train_df = pd.DataFrame({'text': X_train_text, 'label': y_train})
test_df = pd.DataFrame({'text': X_test_text, 'label': y_test})

# Menyimpan data training dan testing ke SQL database
train_df.to_sql(name='train', con=engine, if_exists='replace', index=False)
test_df.to_sql(name='test', con=engine, if_exists='replace', index=False)