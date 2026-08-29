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
    cur.execute("SELECT id, tgl_tweet, full_text FROM ikn")
    data = cur.fetchall()
    cur.close()
    return render_template('dataset.html', ikn=data)


def parseDatasetCSV(filePath):
    print("Memparsing CSV di:", filePath)
    col_names = ['conversation_id_str', 'tgl_tweet', 'fav_count', 'full_text', 'id_str', 'img_url', 
                 'in_reply_to_screen_name', 'lang', 'location', 'quote_count', 'reply_count', 
                 'retweet_count', 'tweet_url', 'user_id_str', 'username']
    try:
        try:
            csvData = pd.read_csv(filePath, names=col_names, header=None, encoding='utf-8')
        except UnicodeDecodeError:
            csvData = pd.read_csv(filePath, names=col_names, header=None, encoding='latin1')
        csvData = csvData.where(pd.notnull(csvData), None)
        print("Data CSV Berhasil Dimuat")
    except Exception as e:
        print("Gagal memuat CSV:", e)
        return

    # Lewati baris header jika ada
    first_row = csvData.iloc[0]
    if str(first_row.get('full_text', '')).lower() in ['full_text', 'tweet', 'text']:
        csvData = csvData.iloc[1:]

    values = []
    for _, row in csvData.iterrows():
        tgl = row.get('tgl_tweet')
        text = row.get('full_text')
        user = row.get('username')
        if text:
            values.append((str(tgl) if tgl else '', str(text), str(user) if user else ''))

    if values:
        sql = """
        INSERT INTO ikn (tgl_tweet, full_text, username) 
        VALUES (%s, %s, %s)
        """
        try:
            cur = mysql.connection.cursor()
            cur.executemany(sql, values)
            mysql.connection.commit()
            cur.close()
            print(f"{len(values)} data berhasil dimasukkan ke tabel ikn")
        except Exception as err:
            print("Kesalahan dalam SQL Insert:", err)
            mysql.connection.rollback()


@app.route('/pelabelan')
def pelabelan():
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, label FROM labelled")
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
    cur.execute("SELECT text, label FROM labelled")
    data = cur.fetchall()
    cur.close()
    return render_template('labelling2.html', ikn_labelled=data)


def parseLabellingCSV(filePath):
    print("Memparsing CSV Label di:", filePath)
    try:
        try:
            csvData = pd.read_csv(filePath, encoding='utf-8')
        except UnicodeDecodeError:
            csvData = pd.read_csv(filePath, encoding='latin1')
        csvData = csvData.where(pd.notnull(csvData), None)
        print("Data CSV Label Berhasil Dimuat")
    except Exception as e:
        print("Gagal memuat CSV Label:", e)
        return

    # Deteksi nama kolom (text, label) atau tanpa header
    cols = [c.lower() for c in csvData.columns]
    values = []
    if 'text' in cols and ('label' in cols or 'label2' in cols):
        label_col = 'label2' if 'label2' in cols else 'label'
        for _, row in csvData.iterrows():
            txt = row.get('text')
            lbl = row.get(label_col)
            if txt:
                values.append((str(txt), str(lbl) if lbl else 'Netral'))
    else:
        # Fallback baca ulang tanpa header
        try:
            try:
                raw_data = pd.read_csv(filePath, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                raw_data = pd.read_csv(filePath, header=None, encoding='latin1')
            for _, row in raw_data.iterrows():
                txt = row[0]
                lbl = row[1] if len(row) > 1 else 'Netral'
                if txt and str(txt).lower() not in ['text', 'tweet']:
                    values.append((str(txt), str(lbl)))
        except Exception as e:
            print("Gagal memproses baris CSV Label:", e)

    if values:
        sql = """
        INSERT INTO labelled (text, label) 
        VALUES (%s, %s)
        """
        try:
            cur = mysql.connection.cursor()
            cur.executemany(sql, values)
            mysql.connection.commit()
            cur.close()
            print(f"{len(values)} data label berhasil dimasukkan")
        except Exception as err:
            print("Kesalahan dalam SQL Insert Label:", err)
            mysql.connection.rollback()


@app.route('/preprocessing')
def preprocessing():
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, text_clean, token, normalisasi, stopwords, stemming, label from preprocessing")
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
    cur.execute("SELECT text, label from train")
    data = cur.fetchall()
    cur.close()
    return render_template('train.html', ikn_train=data)


@app.route('/test')
def test():
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, label from test")
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
        cur.execute("SELECT Text, true_label, predicted_label from klasifikasi")
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
