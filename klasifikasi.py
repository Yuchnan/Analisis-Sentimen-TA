from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

# Koneksi ke MySQL
engine = create_engine("mysql+pymysql://root:@localhost/ta2")

# Memuat dataset pelatihan dan pengujian dari database MySQL
train_df = pd.read_sql_table('train', con=engine)
test_df = pd.read_sql_table('test', con=engine)

# Memisahkan teks dan label
X_train_text = train_df['text']
y_train = train_df['label']
X_test_text = test_df['text']
y_test = test_df['label']

# Mentransformasi data teks menggunakan TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8', ngram_range=(1, 1), max_features=5000)
X_train = tfidf_vectorizer.fit_transform(X_train_text)
X_test = tfidf_vectorizer.transform(X_test_text)

# Mendapatkan fitur (vocabulary)
features = tfidf_vectorizer.get_feature_names_out()

tf = X_train.toarray()
idf = tfidf_vectorizer.idf_
tfidf = tf * idf
df_tf = pd.DataFrame(tf, columns=features)
df_idf = pd.DataFrame(idf.reshape(1, -1), columns=features)
df_tfidf = pd.DataFrame(tfidf, columns=features)

df_tf.to_csv('tfidf/tf.csv')
df_idf.to_csv('tfidf/idf.csv')
df_tfidf.to_csv('tfidf/tfidf.csv')

# Fungsi pembantu untuk menghitung probabilitas prior kelas
def calculate_prior(y):
    classes, counts = np.unique(y, return_counts=True)
    total_count = len(y)
    priors = {cls: count / total_count for cls, count in zip(classes, counts)}
    return priors

# Mengecek distribusi kelas dalam data pelatihan
print("Distribusi kelas dalam data pelatihan:")
print(y_train.value_counts())

# Menghitung prior probabilities
priors = calculate_prior(y_train)
print("Prior Probabilities:")
for cls, prior in priors.items():
    print(f"Kelas '{cls}': {prior:.4f}")

# Likelihood menggunakan Laplace smoothing
def calculate_likelihood(X, y, classes, alpha=1):
    vocab_size = X.shape[1]
    likelihood = {}
    for cls in classes:
        X_cls = X[y == cls]
        word_count = np.sum(X_cls, axis=0) + alpha  # Laplace smoothing
        likelihood[cls] = word_count / (np.sum(word_count) + alpha * vocab_size)
    return likelihood

# Menghitung likelihood
likelihood = calculate_likelihood(X_train.toarray(), y_train, classes=np.unique(y_train))
print("Likelihood untuk setiap kelas:")
for cls in np.unique(y_train):
    print(f"Kelas '{cls}':")
    print(likelihood[cls])

# Fungsi untuk menghitung probabilitas posterior dan memprediksi kelas untuk satu dokumen
def predict_document(doc, priors, likelihood, classes):
    log_probs = {}
    for cls in classes:
        log_prob = np.log(priors[cls]) + doc @ np.log(likelihood[cls])
        log_probs[cls] = log_prob
    return max(log_probs, key=log_probs.get)

# Melatih model Naive Bayes secara manual
classes = np.unique(y_train)
priors = calculate_prior(y_train)
likelihood = calculate_likelihood(X_train.toarray(), y_train, classes, alpha=1)

# Memprediksi label set pengujian secara manual
y_pred_manual = np.array([predict_document(doc, priors, likelihood, classes) for doc in X_test.toarray()])

# Menghitung dan mencetak akurasi
accuracy = accuracy_score(y_test, y_pred_manual)

# Menghitung matriks kebingungan (confusion matrix)
conf_matrix = confusion_matrix(y_test, y_pred_manual)

# Menghitung precision, recall, dan f1 score
precision = precision_score(y_test, y_pred_manual, average='macro')
recall = recall_score(y_test, y_pred_manual, average='macro')
f1 = f1_score(y_test, y_pred_manual, average='macro')

labels = ["Negatif", "Netral", "Positif"]

# Membuat dataframe dengan label asli, label prediksi, dan teks
results_df = pd.DataFrame({
    'Text': X_test_text,
    'true_label': y_test,
    'predicted_label': y_pred_manual
})

# Menyimpan hasil ke database dan file CSV
results_df.to_sql(name='klasifikasi', con=engine, if_exists='replace', index=False)
results_df.to_csv('ikn_final.csv')

# Mencetak matriks kebingungan dalam format DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
print("Confusion Matrix:")
print(conf_matrix_df)
conf_matrix_df.to_csv('confusionmatrix.csv')

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Memeriksa prediksi untuk beberapa sampel
# print("Prediksi untuk beberapa sampel:")
# for i in range(5):  # Memeriksa 5 sampel pertama
#     doc = X_test.toarray()[i]
#     pred = predict_document(doc, priors, likelihood, classes)
#     print(f"Teks: {X_test_text.iloc[i]}")
#     print(f"Label Asli: {y_test.iloc[i]}")
#     print(f"Label Prediksi: {pred}")
