from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
import os
import sys
import pandas as pd
import numpy as np
from subprocess import call
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

app.secret_key = 'many random bytes'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ta2'

mysql = MySQL(app)

# mengaktifkan auto reload
app.config["TEMPLATES_AUTO_RELOAD"] = True
# mengaktifkan mode debug
app.config["DEBUG"] = True
# folder untuk upload file
UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER2 = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/pics', exist_ok=True)
os.makedirs('tfidf', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2


@app.route('/')
def Index():
    return render_template('landing.html')


@app.route('/dataset', methods=['GET', 'POST'])
def Dataset():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename != '':
            if uploaded_file.filename.lower().endswith('.csv'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(file_path)
                parseDatasetCSV(file_path)
                return redirect(url_for("Dataset"))
            else:
                return "File bukan CSV", 400
        return "Tidak ada file yang diupload", 400

    cur = mysql.connection.cursor()
    cur.execute("SELECT id, tgl_tweet, full_text FROM ikn WHERE full_text NOT IN ('full_text', 'tweet', 'text')")
    data = cur.fetchall()
    cur.close()
    return render_template('dataset.html', ikn=data)


def parseDatasetCSV(filePath):
    print("Memparsing CSV di:", filePath)
    HEADER_KEYWORDS = {'full_text', 'tweet', 'text', 'created_at', 'tgl_tweet', 'conversation_id_str', 'username', 'content'}

    try:
        # Coba baca 5 baris pertama untuk memeriksa header
        try:
            sample_df = pd.read_csv(filePath, nrows=5, encoding='utf-8')
        except UnicodeDecodeError:
            sample_df = pd.read_csv(filePath, nrows=5, encoding='latin1')
        
        cols_lower = [str(c).strip().lower() for c in sample_df.columns]
        has_header = any(c in HEADER_KEYWORDS for c in cols_lower)

        if has_header:
            try:
                csvData = pd.read_csv(filePath, encoding='utf-8')
            except UnicodeDecodeError:
                csvData = pd.read_csv(filePath, encoding='latin1')
            
            # Cari kolom teks, tanggal, username
            text_col = next((c for c in csvData.columns if str(c).strip().lower() in ['full_text', 'tweet', 'text', 'content']), None)
            date_col = next((c for c in csvData.columns if str(c).strip().lower() in ['tgl_tweet', 'created_at', 'date', 'tanggal']), None)
            user_col = next((c for c in csvData.columns if str(c).strip().lower() in ['username', 'user', 'screen_name']), None)

            if not text_col:
                text_col = csvData.columns[3] if len(csvData.columns) > 3 else csvData.columns[0]
            if not date_col and len(csvData.columns) > 1:
                date_col = csvData.columns[1]
            if not user_col and len(csvData.columns) > 14:
                user_col = csvData.columns[14]

            values = []
            for _, row in csvData.iterrows():
                txt = str(row.get(text_col, '')).strip() if pd.notnull(row.get(text_col)) else ''
                tgl = str(row.get(date_col, '')).strip() if date_col and pd.notnull(row.get(date_col)) else ''
                usr = str(row.get(user_col, '')).strip() if user_col and pd.notnull(row.get(user_col)) else ''
                
                # Jangan masukkan jika baris ini adalah header
                if txt and txt.lower() not in HEADER_KEYWORDS:
                    if tgl.lower() not in HEADER_KEYWORDS:
                        values.append((tgl, txt, usr))
        else:
            col_names = ['conversation_id_str', 'tgl_tweet', 'fav_count', 'full_text', 'id_str', 'img_url', 
                         'in_reply_to_screen_name', 'lang', 'location', 'quote_count', 'reply_count', 
                         'retweet_count', 'tweet_url', 'user_id_str', 'username']
            try:
                csvData = pd.read_csv(filePath, names=col_names, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                csvData = pd.read_csv(filePath, names=col_names, header=None, encoding='latin1')
            
            values = []
            for _, row in csvData.iterrows():
                tgl = str(row.get('tgl_tweet', '')).strip() if pd.notnull(row.get('tgl_tweet')) else ''
                txt = str(row.get('full_text', '')).strip() if pd.notnull(row.get('full_text')) else ''
                usr = str(row.get('username', '')).strip() if pd.notnull(row.get('username')) else ''
                
                if txt and txt.lower() not in HEADER_KEYWORDS:
                    if tgl.lower() not in HEADER_KEYWORDS:
                        values.append((tgl, txt, usr))

        if values:
            sql = """
            INSERT INTO ikn (tgl_tweet, full_text, username) 
            VALUES (%s, %s, %s)
            """
            cur = mysql.connection.cursor()
            cur.executemany(sql, values)
            mysql.connection.commit()
            cur.close()
            print(f"{len(values)} data berhasil dimasukkan ke tabel ikn (baris header diabaikan).")
    except Exception as e:
        print("Gagal memuat CSV:", e)


@app.route('/pelabelan')
def pelabelan():
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, label FROM labelled WHERE text NOT IN ('full_text', 'tweet', 'text', 'created_at')")
    data = cur.fetchall()
    cur.close()
    return render_template('labelling.html', ikn_labelled=data)


@app.route('/labelling')
def labelling():
    call([sys.executable, "pelabelan.py"])
    return redirect(url_for('pelabelan'))


@app.route('/pelabelan2', methods=['GET', 'POST'])
@app.route('/labelling2', methods=['GET', 'POST'])
def pelabelan2():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename != '':
            if uploaded_file.filename.lower().endswith('.csv'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER2'], uploaded_file.filename)
                uploaded_file.save(file_path)
                parseLabellingCSV(file_path)
                return redirect(url_for("pelabelan2"))
            else:
                return "File bukan CSV", 400
        return "Tidak ada file yang diupload", 400

    cur = mysql.connection.cursor()
    cur.execute("SELECT text, label FROM labelled WHERE text NOT IN ('full_text', 'tweet', 'text', 'created_at')")
    data = cur.fetchall()
    cur.close()
    return render_template('labelling2.html', ikn_labelled=data)


def parseLabellingCSV(filePath):
    print("Memparsing CSV Label di:", filePath)
    HEADER_KEYWORDS = {'full_text', 'tweet', 'text', 'created_at', 'label', 'label2', 'sentiment', 'sentimen'}
    try:
        try:
            csvData = pd.read_csv(filePath, encoding='utf-8')
        except UnicodeDecodeError:
            csvData = pd.read_csv(filePath, encoding='latin1')
        csvData = csvData.where(pd.notnull(csvData), None)
        
        cols = [str(c).strip().lower() for c in csvData.columns]
        values = []
        if 'text' in cols and ('label' in cols or 'label2' in cols):
            label_col = 'label2' if 'label2' in cols else 'label'
            for _, row in csvData.iterrows():
                txt = str(row.get('text', '')).strip()
                lbl = str(row.get(label_col, '')).strip()
                if txt and txt.lower() not in HEADER_KEYWORDS:
                    values.append((txt, lbl if lbl else 'Netral'))
        else:
            try:
                try:
                    raw_data = pd.read_csv(filePath, header=None, encoding='utf-8')
                except UnicodeDecodeError:
                    raw_data = pd.read_csv(filePath, header=None, encoding='latin1')
                for _, row in raw_data.iterrows():
                    txt = str(row[0]).strip() if pd.notnull(row[0]) else ''
                    lbl = str(row[1]).strip() if len(row) > 1 and pd.notnull(row[1]) else 'Netral'
                    if txt and txt.lower() not in HEADER_KEYWORDS:
                        values.append((txt, lbl))
            except Exception as e:
                print("Gagal membaca CSV Label raw:", e)

        if values:
            sql = """
            INSERT INTO labelled (text, label) 
            VALUES (%s, %s)
            """
            cur = mysql.connection.cursor()
            cur.executemany(sql, values)
            mysql.connection.commit()
            cur.close()
            print(f"{len(values)} data label berhasil dimasukkan (baris header diabaikan).")
    except Exception as e:
        print("Gagal memuat CSV Label:", e)


@app.route('/preprocessing')
def preprocessing():
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, text_clean, token, normalisasi, stopwords, stemming, label from preprocessing WHERE text NOT IN ('full_text', 'tweet', 'text', 'created_at')")
    data = cur.fetchall()
    cur.close()
    return render_template('preprocessing.html', ikn_prepro=data)


@app.route('/preprocess')
def preprocess():
    call([sys.executable, "preprocessing.py"])
    call([sys.executable, "split_data.py"])
    return redirect(url_for('preprocessing'))


@app.route('/train')
def train():
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, label from train WHERE text NOT IN ('full_text', 'tweet', 'text', 'created_at')")
    data = cur.fetchall()
    cur.close()
    return render_template('train.html', ikn_train=data)


@app.route('/test')
def test():
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, label from test WHERE text NOT IN ('full_text', 'tweet', 'text', 'created_at')")
    data = cur.fetchall()
    cur.close()
    return render_template('test.html', ikn_test=data)


@app.route('/visualisasi')
def visualisasi():
    pics_dir = 'static/pics'
    files = os.listdir(pics_dir) if os.path.exists(pics_dir) else []
    wordcloud1 = 'pics/wordcloud_positif.png' if 'wordcloud_positif.png' in files else None
    wordcloud2 = 'pics/wordcloud_negatif.png' if 'wordcloud_negatif.png' in files else None
    wordcloud3 = 'pics/wordcloud_all.png' if 'wordcloud_all.png' in files else None
    perbandingan = 'pics/perbandingan.png' if 'perbandingan.png' in files else None
    perbandingan2 = 'pics/perbandingan2.png' if 'perbandingan2.png' in files else None

    return render_template(
        'visual.html',
        wordcloud1=wordcloud1,
        wordcloud2=wordcloud2,
        wordcloud3=wordcloud3,
        perbandingan=perbandingan,
        perbandingan2=perbandingan2
    )


@app.route('/evaluasi')
def evaluasi():
    labels = ["Negatif", "Netral", "Positif"]
    data = []
    akurasi = 0.0
    precision = 0.0
    recall = 0.0
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT Text, true_label, predicted_label from klasifikasi WHERE Text NOT IN ('full_text', 'tweet', 'text', 'created_at')")
        data = cur.fetchall()
        cur.close()

        if data:
            y_true = [row[1] for row in data]
            y_pred = [row[2] for row in data]
            akurasi = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    except Exception as e:
        print("Error saat membaca data evaluasi:", e)

    return render_template(
        'evaluasi.html',
        enumerate=enumerate,
        matrix=matrix,
        labels=labels,
        ikn_klasifikasi=data,
        akurasi=akurasi,
        recall=recall,
        precision=precision
    )


@app.route('/klasifikasi')
def klasifikasi():
    call([sys.executable, "klasifikasi.py"])
    call([sys.executable, "visualisasi.py"])
    return redirect(url_for('evaluasi'))


@app.route('/deleteikn')
def Deleteikn():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE ikn")
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('Dataset'))


@app.route('/deleteikn2')
def Deleteikn2():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE preprocessing")
    cur.execute("TRUNCATE TABLE train")
    cur.execute("TRUNCATE TABLE test")
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('preprocessing'))


@app.route('/deletelabel')
def Deletesplit():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE table labelled")
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('pelabelan'))


@app.route('/deletenb')
def Deletenb():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE klasifikasi")
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('evaluasi'))


@app.route('/deleteall')
def Deleteall():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE ikn")
    cur.execute("TRUNCATE TABLE klasifikasi")
    cur.execute("TRUNCATE TABLE preprocessing")
    cur.execute("TRUNCATE TABLE labelled")
    cur.execute("TRUNCATE TABLE train")
    cur.execute("TRUNCATE TABLE test")
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('Index'))


if __name__ == "__main__":
    app.run(debug=True)
