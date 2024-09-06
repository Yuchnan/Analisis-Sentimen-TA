import os
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

# Membuat direktori untuk menyimpan gambar jika belum ada
os.makedirs("static/pics", exist_ok=True)

# Membaca data dari CSV
df = pd.read_csv("ikn_final.csv", usecols=['Text', 'true_label', 'predicted_label'])

# Memisahkan data berdasarkan sentimen
df_positif = df[df['true_label'] == "Positif"]
df_negatif = df[df['true_label'] == "Negatif"]
df_netral = df[df['true_label'] == "Netral"]

# Membuat WordCloud untuk sentimen negatif
all_text_s0 = ' '.join(word for word in df_negatif["Text"])
wordcloud_negatif = WordCloud(colormap='Reds', width=3000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
plt.figure(figsize=(9, 3))
plt.imshow(wordcloud_negatif, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.savefig("static/pics/wordcloud_negatif.png")

# Membuat WordCloud untuk sentimen positif
all_text_s1 = ' '.join(word for word in df_positif["Text"])
wordcloud_positif = WordCloud(colormap='Blues', width=3000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
plt.figure(figsize=(9, 3))
plt.imshow(wordcloud_positif, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.savefig("static/pics/wordcloud_positif.png")

# Menampilkan jumlah data masing-masing sentimen
true_label_counts = df['true_label'].value_counts()
predicted_label_counts = df['predicted_label'].value_counts()

# Fungsi untuk menambahkan nilai di bawah batang
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart."""
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing
        va = 'bottom'

        label = "{:.0f}".format(y_value)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va
        )

# Visualisasi perbandingan sentimen positif, negatif, dan netral (true_label)
plt.figure(figsize=(7, 4))
ax1 = plt.bar(true_label_counts.index, true_label_counts.values, color=['#FF9999', '#66B2FF', '#99FF99'])
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')
plt.title('Perbandingan Sentimen (True Label)')
add_value_labels(plt.gca(), spacing=-20)  # Tambahkan nilai di bawah batang
plt.savefig("static/pics/perbandingan.png")

# Visualisasi perbandingan sentimen positif, negatif, dan netral (predicted_label)
plt.figure(figsize=(7, 4))
ax2 = plt.bar(predicted_label_counts.index, predicted_label_counts.values, color=['#FF9999', '#66B2FF', '#99FF99'])
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')
plt.title('Perbandingan Sentimen (Predicted Label)')
add_value_labels(plt.gca(), spacing=-20)  # Tambahkan nilai di bawah batang
plt.savefig("static/pics/perbandingan2.png")

all_text = ' '.join(df["Text"])

# Fungsi untuk menampilkan n kata paling sering
def top_n_words(text, n=5):
    word_freq = pd.Series(text.split()).value_counts()
    return word_freq.head(n)

# Menampilkan 5 kata paling sering secara keseluruhan
top_words_all = top_n_words(all_text)

# Menyimpan grafik kata paling sering
plt.figure(figsize=(7, 4))
top_words_all.plot(kind='barh', color='purple')
plt.xlabel('Frekuensi')
plt.title('5 Kata Paling Sering')
plt.savefig("static/pics/top_words_all.png")

# Membuat WordCloud untuk semua teks
wordcloud_all = WordCloud(colormap='viridis', width=3000, height=1000, mode='RGBA', background_color='white').generate(all_text)
plt.figure(figsize=(9, 3))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.savefig("static/pics/wordcloud_all.png")
