# Web Aplikasi Analisis Sentimen (Multinomial Naïve Bayes)

Aplikasi web analisis sentimen opini masyarakat terhadap pemindahan Ibu Kota Negara (IKN) Nusantara menggunakan algoritma **Multinomial Naïve Bayes** berbasis **Flask** dan **MySQL**.

Perhitungan probabilitas **Prior**, **Likelihood** (dengan *Laplace Smoothing*), dan **Posterior Log-Probability** pada algoritma Naïve Bayes diimplementasikan secara terstruktur dan mandiri (*from scratch*) untuk memprediksi label sentimen: **Positif**, **Netral**, dan **Negatif**.

---

## 🌟 Fitur Utama

1. **Import Dataset**: Unggah data tweet hasil crawling (.csv) dengan penyimpanan ke basis data MySQL.
2. **Pelabelan Otomatis (Lexicon-based)**: Penentuan label sentimen awal (Positif, Negatif, Netral) secara otomatis menggunakan kamus leksikon sentimen bahasa Indonesia.
3. **Import Data Label**: Fleksibilitas untuk mengunggah dataset yang telah dilabeli manual (.csv).
4. **Text Preprocessing**:
   - *Case Folding* (mengubah teks menjadi huruf kecil)
   - *Cleansing* (pembersihan URL, mention, hashtag, karakter khusus, dan angka)
   - *Tokenizing* (pemotongan kalimat menjadi token kata dengan NLTK)
   - *Normalization* (konversi kata slang/tidak baku menjadi baku)
   - *Stopwords Removal* (penghapusan kata-kata umum yang tidak berpengaruh)
   - *Stemming* (pengembalian kata berimbuhan ke kata dasar menggunakan PySastrawi)
5. **Data Splitting**: Pembagian data latih (*training*) dan data uji (*testing*) dengan rasio 80:20 (Stratified Sampling).
6. **Klasifikasi & Evaluasi**:
   - Pembobotan kata menggunakan **TF-IDF**.
   - Klasifikasi menggunakan algoritma **Multinomial Naïve Bayes**.
   - Perhitungan metrik evaluasi: **Accuracy**, **Precision**, **Recall**, **F1-Score**, serta tabel **Confusion Matrix**.
7. **Visualisasi Data**:
   - Diagram batang perbandingan distribusi sentimen asli (*True Label*) dan hasil prediksi (*Predicted Label*).
   - *WordCloud* untuk sentimen Positif, Negatif, dan Keseluruhan.
   - Grafik 5 kata dengan frekuensi kemunculan tertinggi (*Top Words*).
8. **Manajemen Data**: Tombol pembersihan (*truncate*) data per tahap maupun reset seluruh dataset secara instan.

---

## 🛠️ Teknologi & Library

- **Backend**: Python 3.10+, Flask, Flask-MySQLdb, SQLAlchemy, PyMySQL
- **Database**: MySQL / MariaDB (XAMPP)
- **NLP & Machine Learning**: NLTK, PySastrawi, Scikit-Learn (TF-IDF & Metrics), Pandas, NumPy
- **Visualisasi**: Matplotlib, WordCloud
- **Frontend**: HTML5, Bootstrap 4/5, SB Admin 2, DataTables, FontAwesome

---

## 📋 Struktur Direktori

```plaintext
Analisis-Sentimen-TA/
├── kamus/                     # Kamus leksikon sentimen, slang, dan stopwords
│   ├── positive.tsv
│   ├── negative.tsv
│   ├── slang.csv
│   └── stopwords.txt
├── static/                    # Asset CSS, JS, Vendor, dan gambar hasil visualisasi
│   ├── css/
│   ├── js/
│   ├── pics/                  # Output grafik dan WordCloud
│   └── vendor/
├── templates/                 # Template HTML (Jinja2)
│   ├── landing.html           # Halaman utama / beranda
│   ├── dataset.html           # Halaman impor dataset
│   ├── labelling.html         # Halaman pelabelan otomatis
│   ├── labelling2.html        # Halaman impor data berlabel
│   ├── preprocessing.html     # Halaman proses preprocessing
│   ├── train.html             # Halaman data latih
│   ├── test.html              # Halaman data uji
│   ├── evaluasi.html          # Halaman hasil evaluasi & confusion matrix
│   └── visual.html            # Halaman visualisasi grafik & wordcloud
├── tfidf/                     # Penyimpanan matriks TF, IDF, dan TF-IDF (CSV)
├── uploads/                   # Penyimpanan file CSV yang diunggah
├── klasifikasi.py             # Script model Multinomial Naive Bayes & TF-IDF
├── pelabelan.py               # Script pelabelan otomatis berbasis leksikon
├── preprocessing.py           # Script tahapan text preprocessing
├── split_data.py              # Script pembagian data latih dan data uji
├── visualisasi.py             # Script pembuatan visualisasi grafik & wordcloud
├── main.py                    # Entry point aplikasi web Flask & routing
├── ta2.sql                    # Skema database MySQL
├── requirements.txt           # Daftar dependensi Python
└── README.md                  # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Menjalankan Aplikasi

### 1. Prasyarat
- Python 3.10 atau versi lebih baru.
- XAMPP / MySQL Server.
- Git (opsional).

### 2. Konfigurasi Basis Data
1. Jalankan service **Apache** dan **MySQL** melalui XAMPP Control Panel.
2. Buka phpMyAdmin di browser (`http://localhost/phpmyadmin`).
3. Buat database baru dengan nama `ta2`.
4. Import file `ta2.sql` yang ada di root direktori proyek ini ke dalam database `ta2`.

### 3. Persiapan Lingkungan Python & Dependensi
Buka terminal/PowerShell di direktori proyek, lalu jalankan:

```bash
# (Opsional) Buat dan aktifkan virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependensi yang dibutuhkan
pip install -r requirements.txt
```

### 4. Mengunduh Resource NLTK
Jalankan perintah berikut untuk memastikan tokenizers dan stopwords NLTK terunduh:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 5. Menjalankan Aplikasi Web
Jalankan server Flask dengan perintah:
```bash
python main.py
```
Aplikasi akan berjalan di:
👉 **`http://127.0.0.1:5000`**

Buka URL tersebut di peramban web (Chrome / Edge / Firefox).

---

## 📖 Alur Penggunaan Aplikasi

1. **Beranda**: Halaman pembuka informasi tugas akhir.
2. **Import Dataset**: Unggah file CSV tweet mentah di menu *Master Data -> Import Dataset*.
3. **Pelabelan**: Klik tombol *Pelabelan* untuk melabeli tweet secara otomatis berdasarkan kamus leksikon atau gunakan *Import Data Label* jika memiliki label sendiri.
4. **Preprocessing**: Buka menu *Preprocessing* dan klik tombol *Preprocessing* untuk menjalankan tahapan pembersihan teks dan pembagian data train/test secara otomatis.
5. **Data Latih & Data Uji**: Tinjau hasil data yang telah dibagi pada menu *Data Latih* dan *Data Uji*.
6. **Evaluasi**: Buka menu *Evaluasi* dan klik tombol *Klasifikasi* untuk melatih model Naïve Bayes, melihat hasil prediksi, akurasi, presisi, recall, dan confusion matrix.
7. **Visualisasi Data**: Buka menu *Visualisasi Data* untuk melihat WordCloud dan grafik perbandingan sentimen.

---

## 👤 Pengembang
- **Nama**: Ramandhanu Yuchnan Utomo
- **NIM**: 2011500333
