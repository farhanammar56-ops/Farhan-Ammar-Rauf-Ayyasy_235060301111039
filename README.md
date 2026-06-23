#Topil : Kesehatan

# Evaluasi Penanganan Penderita Diabetes Berdasarkan Parameter HbA1c dan Obat-Obatan yang Diterima

Proyek ini merupakan implementasi Machine Learning menggunakan algoritma **Random Forest Classifier** untuk memprediksi rekomendasi penanganan medis pada penderita diabetes. Prediksi didasarkan pada hasil tes kadar **HbA1c** (kolom `A1Cresult`), profil demografi, serta status riwayat konsumsi obat-obatan pasien.

Aplikasi ini dilengkapi dengan antarmuka interaktif berbasis **Gradio** untuk memudahkan tenaga medis melakukan simulasi prediksi secara *real-time*.

Dataset yang digunakan berasal dari data historis **130 Rumah Sakit di Amerika Serikat selama 10 tahun (1999-2008)** (`diabetic_data.csv`). Link dataset : https://doi.org/10.24432/C5230J
* **Kunci Utama**: `A1Cresult` (Hasil tes HbA1c pasien).
* **Target Prediksi**: `change` (Menunjukkan apakah diperlukan perubahan penanganan/dosis obat diabetes: `Ch` atau `No`).
* **Fitur Pendukung**: Jenis kelamin, usia, ras, durasi rawat inap, jumlah prosedur laboratorium, serta status 4 jenis obat utama (`metformin`, `glipizide`, `glyburide`, `insulin`).

---

##Tahapan Proyek

### 1. Preprocessing Data
* Mengubah nilai corrupt `?` menjadi format `NaN` resmi untuk pembersihan data.
* Mengisi missing value pada `A1Cresult` dengan string `'None'` karena nilai kosong secara medis menandakan pasien tidak melalui tes HbA1c saat dirawat.
* Seleksi fitur relevan guna mereduksi dimensi data yang tidak perlu (seperti kolom `weight` yang memiliki eror/kosong sebesar 96%).
* Transformasi data kategorikal menjadi numerik menggunakan `LabelEncoder` dan pemetaan manual terstruktur.
* Pembagian data menggunakan `train_test_split` dengan rasio **80% Data Training** dan **20% Data Testing**.

### 2. Training & Testing Model
Model dilatih menggunakan algoritma **Random Forest Classifier** dari pustaka `scikit-learn` untuk menangani klasifikasi data tabel tabular berskala besar secara optimal.

### 3. Evaluasi Model
Model diuji menggunakan data testing independen dan menghasilkan performa tinggi sebagai berikut:
* **Akurasi**: ~98.4%
* **Confusion Matrix**: Menunjukkan tingkat *True Positive* dan *True Negative* yang presisi tinggi dengan angka eror prediksi (*False Positive* / *False Negative*) yang sangat minim.

---

Detail Petunjuk dalam bentuk pdf termasuk Confusion Matrix: [Farhan.Ammar.Rauf.Ayyasy_235060301111039_UAS.Kecerdasan.Buatan.pdf](https://github.com/user-attachments/files/29252246/Farhan.Ammar.Rauf.Ayyasy_235060301111039_UAS.Kecerdasan.Buatan.pdf)

Link Video Demo dan Slide : 
