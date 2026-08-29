import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Koneksi ke MySQL
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

# Folder output tfidf
os.makedirs('tfidf', exist_ok=True)

accuracy = 0.0
precision = 0.0
recall = 0.0
f1 = 0.0
labels = ['Negatif', 'Netral', 'Positif']
conf_matrix = np.zeros((3, 3))

def calculate_prior(y, power=0.5):
    classes, counts = np.unique(y, return_counts=True)
    total_count = len(y)
    raw_priors = {cls: count / total_count for cls, count in zip(classes, counts)}
    # Power-scaled priors to prevent class imbalance collapse
    priors = {cls: raw_priors[cls] ** power for cls in classes}
    norm_sum = sum(priors.values())
    return {cls: v / norm_sum for cls, v in priors.items()}

def calculate_likelihood(X, y, classes, alpha=0.05):
    vocab_size = X.shape[1]
    likelihood = {}
    for cls in classes:
        X_cls = X[y == cls]
        word_count = np.sum(X_cls, axis=0) + alpha  # Laplace smoothing
        likelihood[cls] = word_count / (np.sum(word_count) + alpha * vocab_size)
    return likelihood

def predict_document(doc, priors, likelihood, classes):
    log_probs = {}
    for cls in classes:
        p_cls = priors.get(cls, 1e-9)
        l_cls = np.where(likelihood[cls] <= 0, 1e-9, likelihood[cls])
        log_prob = np.log(p_cls) + doc @ np.log(l_cls)
        log_probs[cls] = log_prob
    return max(log_probs, key=log_probs.get)

def run_classification():
    global accuracy, precision, recall, f1, conf_matrix, labels
    try:
        train_df = pd.read_sql_table('train', con=engine)
        test_df = pd.read_sql_table('test', con=engine)
    except Exception as e:
        print("Tabel train atau test belum ada di database:", e)
        return

    if train_df.empty or test_df.empty:
        print("Data train atau test kosong!")
        return

    # Memisahkan teks dan label
    X_train_text = train_df['text'].fillna('').astype(str)
    y_train = train_df['label'].fillna('Netral').astype(str)
    X_test_text = test_df['text'].fillna('').astype(str)
    y_test = test_df['label'].fillna('Netral').astype(str)

    # Mentransformasi data teks menggunakan TfidfVectorizer dengan unigram & bigram
    tfidf_vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8', ngram_range=(1, 2), max_features=4000, sublinear_tf=True, min_df=1)
    X_train_sp = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_sp = tfidf_vectorizer.transform(X_test_text)

    # Mendapatkan fitur (vocabulary)
    features = tfidf_vectorizer.get_feature_names_out()

    # Baca kamus untuk memberi bobot fitur sentimen
    pos_file = "kamus/positive.tsv"
    neg_file = "kamus/negative.tsv"
    pos_set = set(pd.read_csv(pos_file, sep="\t", header=None)[0].astype(str).str.lower().str.strip()) if os.path.exists(pos_file) else set()
    neg_set = set(pd.read_csv(neg_file, sep="\t", header=None)[0].astype(str).str.lower().str.strip()) if os.path.exists(neg_file) else set()
    sentiment_lexicon = pos_set | neg_set

    # Matriks bobot penguat kata sentimen
    feature_weights = np.ones(len(features))
    for idx, f in enumerate(features):
        if f in sentiment_lexicon:
            feature_weights[idx] = 2.5

    X_train = X_train_sp.toarray() * feature_weights
    X_test = X_test_sp.toarray() * feature_weights

    tf = X_train_sp.toarray()
    idf = tfidf_vectorizer.idf_
    tfidf = tf * idf
    df_tf = pd.DataFrame(tf, columns=features)
    df_idf = pd.DataFrame(idf.reshape(1, -1), columns=features)
    df_tfidf = pd.DataFrame(tfidf, columns=features)

    df_tf.to_csv('tfidf/tf.csv', index=False)
    df_idf.to_csv('tfidf/idf.csv', index=False)
    df_tfidf.to_csv('tfidf/tfidf.csv', index=False)

    # Training Multinomial Naive Bayes secara manual
    classes = np.unique(y_train)
    priors = calculate_prior(y_train, power=0.5)
    likelihood = calculate_likelihood(X_train, y_train, classes, alpha=0.05)

    print("Distribusi kelas dalam data pelatihan:")
    print(y_train.value_counts())
    print("\nPrior Probabilities (Calibrated):")
    for cls, prior in priors.items():
        print(f"Kelas '{cls}': {prior:.4f}")

    # Memprediksi label set pengujian
    y_pred_manual = np.array([predict_document(doc, priors, likelihood, classes) for doc in X_test])

    # Metrik Evaluasi
    accuracy = accuracy_score(y_test, y_pred_manual)
    conf_matrix = confusion_matrix(y_test, y_pred_manual, labels=labels)
    precision = precision_score(y_test, y_pred_manual, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_manual, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_manual, average='macro', zero_division=0)

    # Membuat dataframe hasil
    results_df = pd.DataFrame({
        'Text': X_test_text,
        'true_label': y_test,
        'predicted_label': y_pred_manual
    })

    # Menyimpan hasil ke database dan CSV
    results_df.to_sql(name='klasifikasi', con=engine, if_exists='replace', index=False)
    results_df.to_csv('ikn_final.csv', index=False)

    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print("\nConfusion Matrix:")
    print(conf_matrix_df)
    conf_matrix_df.to_csv('confusionmatrix.csv')

    print(f"\nPrecision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("Klasifikasi Naive Bayes berhasil diselesaikan!")

# Eksekusi saat file dijalankan langsung atau diimpor jika data tersedia
try:
    run_classification()
except Exception as e:
    print("Error saat menjalankan klasifikasi:", e)
