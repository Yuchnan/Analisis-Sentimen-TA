import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from sqlalchemy import create_engine

# Membuat direktori untuk menyimpan gambar jika belum ada
os.makedirs("static/pics", exist_ok=True)

# Koneksi ke MySQL / fallback ke CSV
try:
    engine = create_engine("mysql+pymysql://root:@localhost/ta2")
    df = pd.read_sql_table('klasifikasi', con=engine)
except Exception:
    if os.path.exists("ikn_final.csv"):
        df = pd.read_csv("ikn_final.csv")
    else:
        df = pd.DataFrame(columns=['Text', 'true_label', 'predicted_label'])

if not df.empty and 'Text' in df.columns and 'true_label' in df.columns:
    df['Text'] = df['Text'].fillna('').astype(str)
    
    # Memisahkan data berdasarkan sentimen
    df_positif = df[df['true_label'] == "Positif"]
    df_negatif = df[df['true_label'] == "Negatif"]
    df_netral = df[df['true_label'] == "Netral"]
    
    # Fungsi pembantu untuk membuat WordCloud dengan aman
    def generate_wc(text_series, colormap, output_path):
        all_text = ' '.join(str(word) for word in text_series if str(word).strip())
        if not all_text.strip():
            all_text = "tidak ada data"
        wc = WordCloud(colormap=colormap, width=1200, height=500, mode='RGBA', background_color='white').generate(all_text)
        plt.figure(figsize=(9, 3.5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()

    # WordCloud Negatif
    generate_wc(df_negatif["Text"], 'Reds', "static/pics/wordcloud_negatif.png")

    # WordCloud Positif
    generate_wc(df_positif["Text"], 'Blues', "static/pics/wordcloud_positif.png")

    # WordCloud Keseluruhan
    generate_wc(df["Text"], 'viridis', "static/pics/wordcloud_all.png")

    # Fungsi untuk menambahkan nilai di atas batang
    def add_value_labels(ax, spacing=3):
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            label = f"{int(y_value)}"
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(0, spacing),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontweight='bold'
            )

    # Visualisasi perbandingan true_label
    true_label_counts = df['true_label'].value_counts()
    plt.figure(figsize=(7, 4))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'][:len(true_label_counts)]
    ax1 = plt.bar(true_label_counts.index, true_label_counts.values, color=colors)
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.title('Perbandingan Sentimen (True Label)')
    add_value_labels(plt.gca(), spacing=3)
    plt.tight_layout()
    plt.savefig("static/pics/perbandingan.png", dpi=100)
    plt.close()

    # Visualisasi perbandingan predicted_label
    predicted_label_counts = df['predicted_label'].value_counts()
    plt.figure(figsize=(7, 4))
    colors2 = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'][:len(predicted_label_counts)]
    ax2 = plt.bar(predicted_label_counts.index, predicted_label_counts.values, color=colors2)
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.title('Perbandingan Sentimen (Predicted Label)')
    add_value_labels(plt.gca(), spacing=3)
    plt.tight_layout()
    plt.savefig("static/pics/perbandingan2.png", dpi=100)
    plt.close()

    # Top n kata paling sering
    all_text = ' '.join(df["Text"])
    if all_text.strip():
        word_freq = pd.Series(all_text.split()).value_counts().head(5)
        plt.figure(figsize=(7, 4))
        word_freq.plot(kind='barh', color='#8854d0')
        plt.xlabel('Frekuensi')
        plt.title('5 Kata Paling Sering')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("static/pics/top_words_all.png", dpi=100)
        plt.close()

print("Visualisasi berhasil dibuat!")
