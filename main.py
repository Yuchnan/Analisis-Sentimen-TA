from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
import os
import pandas as pd
import mysql.connector as mysql2
from subprocess import call
from klasifikasi3 import accuracy, precision, recall, conf_matrix, labels
# from preprocessing import label_counts

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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER2 = 'uploads'
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2

mydb = mysql2.connect(
    host="localhost",
    user="root",
    password="",
    database="ta2"
)

@app.route('/')
def Index():
    return render_template('landing.html')

@app.route('/dataset')
def Dataset():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM ikn")
    data = cur.fetchall()
    cur.close()
    return render_template('dataset.html', ikn=data)

@app.route('/preprocessing')
def preprocessing():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * from preprocessing")
    data = cur.fetchall()
    cur.close()
    return render_template('preprocessing.html', ikn_prepro=data)

@app.route('/preprocess')
def preprocess():
    call(["python", "preprocessing.py"])
    call(["python", "split_data.py"])
    return redirect(url_for('preprocessing'))

@app.route('/pelabelan')
def pelabelan():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * from labelled")
    data = cur.fetchall()
    cur.close()
    return render_template('labelling.html', ikn_labelled=data)

@app.route('/pelabelan2')
def pelabelan2():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * from labelled")
    data = cur.fetchall()
    cur.close()
    return render_template('labelling2.html', ikn_labelled=data)

@app.route('/labelling')
def labelling():
    call(["python", "pelabelan.py"])
    return redirect(url_for('pelabelan'))

@app.route('/train')
def train():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * from train")
    data = cur.fetchall()
    cur.close()
    return render_template('train.html', ikn_train=data)

@app.route('/test')
def test():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * from test")
    data = cur.fetchall()
    cur.close()
    return render_template('test.html', ikn_test=data)

@app.route('/visualisasi')
def visualisasi():
    files = os.listdir('static/pics')
    wordcloud1 = 'pics/wordcloud_positif.png' if 'wordcloud_positif.png' in files else None
    wordcloud2 = 'pics/wordcloud_negatif.png' if 'wordcloud_negatif.png' in files else None
    wordcloud3 = 'pics/wordcloud_all.png' if 'wordcloud_all.png' in files else None
    perbandingan = 'pics/perbandingan.png' if 'perbandingan.png' in files else None
    perbandingan2 = 'pics/perbandingan2.png' if 'perbandingan2.png' in files else None

    return render_template('visual.html', wordcloud1=wordcloud1, wordcloud2=wordcloud2, wordcloud3=wordcloud3, perbandingan=perbandingan, perbandingan2=perbandingan2)

# @app.route('/visual')
# def visual():
#     call(["python", "visualisasi.py"])
#     return redirect(url_for('visualisasi'))

@app.route('/evaluasi')
def evaluasi():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * from klasifikasi")
    data = cur.fetchall()
    cur.close()
    conf_matrix_df = conf_matrix.tolist()
    return render_template('evaluasi.html', enumerate=enumerate, matrix=conf_matrix_df, labels=labels, ikn_klasifikasi=data, akurasi=accuracy, recall=recall, precision=precision)

@app.route('/klasifikasi')
def klasifikasi():
    call(["python", "klasifikasi3.py"])
    call(["python", "visualisasi.py"])
    return redirect(url_for('evaluasi'))

@app.route('/deleteikn')
def Deleteikn():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE ikn")
    mysql.connection.commit()
    return redirect(url_for('Dataset'))

@app.route('/deleteikn2')
def Deleteikn2():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE preprocessing")
    cur.execute("TRUNCATE TABLE train")
    cur.execute("TRUNCATE TABLE test")
    mysql.connection.commit()
    return redirect(url_for('preprocessing'))

@app.route('/deletelabel')
def Deletesplit():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE table labelled")
    mysql.connection.commit()
    return redirect(url_for('pelabelan'))

@app.route('/deletenb')
def Deletenb():
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE klasifikasi")
    mysql.connection.commit()
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
    return redirect(url_for('Index'))

@app.route('/dataset', methods=['POST'])
def uploadDatasetFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        if uploaded_file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            parseDatasetCSV(file_path)
            return redirect(url_for("Dataset"))
        else:
            return "File bukan CSV", 400
    return "Tidak ada file yang diupload", 400

def parseDatasetCSV(filePath):
    print("Memparsing CSV di:", filePath)
    try:
        col_names = ['conversation_id_str', 'tgl_tweet', 'fav_count', 'full_text', 'id_str', 'img_url', 'in_reply_to_screen_name', 'lang', 
                     'location', 'quote_count', 'reply_count', 'retweet_count', 'tweet_url', 'user_id_str', 'username']
        csvData = pd.read_csv(filePath, names=col_names, header=None, encoding='unicode_escape')
        csvData = csvData.where(pd.notnull(csvData), None)  # Mengganti NaN dengan None
        print("Data CSV Berhasil Dimuat")
    except Exception as e:
        print("Gagal memuat CSV:", e)
        return

    values = []
    for i, row in csvData.iterrows():
        values.append((row['tgl_tweet'], row['full_text'], row['username']))
    
    sql = """
    INSERT INTO ikn (tgl_tweet, full_text, username) 
    VALUES (%s, %s, %s) 
    ON DUPLICATE KEY UPDATE 
    tgl_tweet = VALUES(tgl_tweet), 
    full_text = VALUES(full_text), 
    username = VALUES(username)
    """
    try:
        cur = mydb.cursor()
        cur.executemany(sql, values)
        
        mydb.commit()
        print("Data berhasil dimasukkan")
    except mysql2.Error as err:
        print("Kesalahan dalam SQL Insert:", err)
        mydb.rollback()

@app.route('/labelling2', methods=['GET', 'POST'])
def uploadLabellingFiles():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename.endswith('.csv'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER2'], uploaded_file.filename)
                uploaded_file.save(file_path)
                parseLabellingCSV(file_path)
                return redirect(url_for("pelabelan2"))
            else:
                return "File bukan CSV", 400
        return "Tidak ada file yang diupload", 400
    return render_template('labelling2.html')

def parseLabellingCSV(filePath):
    print("Memparsing CSV di:", filePath)
    try:
        col_names = ['text', 'label', 'label2']
        csvData = pd.read_csv(filePath, names=col_names, header=None, encoding='unicode_escape')
        csvData = csvData.where(pd.notnull(csvData), None)  # Mengganti NaN dengan None
        print("Data CSV Berhasil Dimuat")
    except Exception as e:
        print("Gagal memuat CSV:", e)
        return

    values = []
    for i, row in csvData.iterrows():
        values.append((row['text'], row['label2']))
    
    sql = """
    INSERT INTO labelled (text, label) 
    VALUES (%s, %s) 
    ON DUPLICATE KEY UPDATE 
    text = VALUES(text), 
    label = VALUES(label)
    """
    try:
        cur = mydb.cursor()
        cur.executemany(sql, values)
        
        mydb.commit()
        print("Data berhasil dimasukkan")
    except mysql2.Error as err:
        print("Kesalahan dalam SQL Insert:", err)
        mydb.rollback()

if __name__ == "__main__":
    app.run(debug=True)
