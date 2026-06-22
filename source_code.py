#Load Dataset dan Operasi Data 
import os  
print(os.getcwd())

#Cari jumlah Missing Value dan persentasenya 
import pandas as pd 
import numpy as np 
df = pd.read_csv('diabetic_data.csv', na_values='?') 
df.head()
missing_data = df.isnull().sum()  
missing_percent = (df.isnull().sum() / len(df)) * 100 
missing_summary = pd.DataFrame({'Jumlah Kosong': missing_data, 'Persentase (%)': missing_percent}) print(missing_summary[missing_summary['Jumlah Kosong'] > 0])

#Seleksi Kolom & Menentukan Target 
fitur_pilihan = [ 'race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'A1Cresult', 'metformin', 'glipizide', 'glyburide', 'insulin', 'change' ] 
df_clean = df[fitur_pilihan].copy() 
df_clean = df_clean.dropna(subset=['race', 'gender']) 
print("Ukuran data setelah seleksi awal:", df_clean.shape)

#Encoding (Ubah Teks Menjadi Angka)
df_clean['change'] = df_clean['change'].map({'Ch': 1, 'No': 0}) 
df_clean['A1Cresult'] = df_clean['A1Cresult'].map({'None': 0, 'Norm': 1, '>7': 2, '>8': 3}) 
from sklearn.preprocessing import LabelEncoder le = LabelEncoder() 
kolom_teks = ['race', 'gender', 'age', 'metformin', 'glipizide', 'glyburide', 'insulin'] for col in kolom_teks: df_clean[col] = le.fit_transform(df_clean[col].astype(str)) 
df_clean.head()

#Split Data (Data Training dan Data Testing)
from sklearn.model_selection import train_test_split 
X = df_clean.drop(columns=['change'])
y = df_clean['change'] 
# Untuk pembagian data: 80% Training, 20% Testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
print(f"Jumlah Data Training: {X_train.shape[0]} baris") 
print(f"Jumlah Data Testing: {X_test.shape[0]} baris") 

#Training model
from sklearn.ensemble import RandomForestClassifier 
import time 
model_ai = RandomForestClassifier(n_estimators=100, random_state=42) 
print("Memulai proses training model AI... Santai dulu sejenak...") 
waktu_mulai = time.time() 
model_ai.fit(X_train, y_train) 
waktu_selesai = time.time() 
print(f"Training SELESAI dalam waktu {waktu_selesai - waktu_mulai:.2f} detik!") 

#Testing Model
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt  
import seaborn as sns  
y_prediksi = model_ai.predict(X_test) 

#Cetak Hasil Evaluasi Model
cm = confusion_matrix(y_test, y_prediksi) 
plt.figure(figsize=(6,5)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 

            xticklabels=['Tebak Tetap', 'Tebak Berubah'], 

            yticklabels=['Asli Tetap', 'Asli Berubah']) 
plt.ylabel('Data Asli Lapangan') 
plt.xlabel('Hasil Tebakan AI') 
plt.title('Confusion Matrix Rekomendasi Diabetes') 
plt.show() 

#Install Gradio Pada Google Colab 
!pip install -q gradio 

#Buat Fungsi Prediksi & Jalankan Laman Demo 
import gradio as gr 
import numpy as np 
def rekomendasi_penanganan(race, gender, age, time_in_hospital, num_lab_procedures,  

                           num_procedures, num_medications, A1Cresult,  

                           metformin, glipizide, glyburide, insulin): 

    try: 

        race_map = {'Caucasian': 0, 'AfricanAmerican': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4, 'Unknown': 5} 

        race_val = race_map.get(race, 0) 

         

         (Female = 0, Male = 1) 

        gender_val = 0 if gender == "Female" else 1 

         

        

        age_map = { 

            "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4, 

            "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9 

        } 

        age_val = age_map.get(age, 5)  

         

        

        a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3} 

        a1c_val = a1c_map.get(A1Cresult, 0) 

         

 

        med_map = {'Down': 0, 'No': 1, 'Steady': 2, 'Up': 3} 

        met_val = med_map.get(metformin, 1) 

        glip_val = med_map.get(glipizide, 1) 

        gly_val = med_map.get(glyburide, 1) 

        ins_val = med_map.get(insulin, 1) 

         

         

        input_data = np.array([[ 

            race_val, gender_val, age_val, int(time_in_hospital),  

            int(num_lab_procedures), int(num_procedures), int(num_medications),  

            a1c_val, met_val, glip_val, gly_val, ins_val 

        ]]) 

         

         

        prediksi = model_ai.predict(input_data) 

         

        

        if prediksi[0] == 1: 

            return "REKOMENDASI: Perlu Perubahan Penanganan / Dosis Obat Diabetes (Ch)." 

        else: 

            return "REKOMENDASI: Penanganan / Dosis Obat Sudah Sesuai, Pertahankan (No)." 

             

    except Exception as e: 

        # Jika masih ada eror, bakal ketahuan erornya di bagian mana 

        return f"Terjadi kesalahan pada input data: {str(e)}" 

demo = gr.Interface( 

    fn=rekomendasi_penanganan, 

    inputs=[ 

        gr.Dropdown(choices=['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'], value='Caucasian', label="Ras Pasien"), 

        gr.Dropdown(choices=['Female', 'Male'], value='Female', label="Jenis Kelamin"), 

        gr.Dropdown(choices=["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], value="[50-60)", label="Umur Pasien"), 

        gr.Slider(minimum=1, maximum=14, step=1, value=3, label="Waktu Dirawat (Hari)"), 

        gr.Slider(minimum=1, maximum=132, step=1, value=40, label="Jumlah Tes Laboratorium"), 

        gr.Slider(minimum=0, maximum=6, step=1, value=1, label="Jumlah Prosedur Medis Lain"), 

        gr.Slider(minimum=1, maximum=81, step=1, value=15, label="Jumlah Konsumsi Obat"), 

        gr.Dropdown(choices=['None', 'Norm', '>7', '>8'], value='None', label="Hasil Tes HbA1c"), 

        gr.Dropdown(choices=['Down', 'No', 'Steady', 'Up'], value='No', label="Status Obat Metformin"), 

        gr.Dropdown(choices=['Down', 'No', 'Steady', 'Up'], value='No', label="Status Obat Glipizide"), 

        gr.Dropdown(choices=['Down', 'No', 'Steady', 'Up'], value='No', label="Status Obat Glyburide"), 

        gr.Dropdown(choices=['Down', 'No', 'Steady', 'Up'], value='No', label="Status Obat Insulin") 

    ], 

    outputs=gr.Textbox(label="Hasil Analisis AI"), 

    title="AI Rekomendasi Penanganan Diabetes Berdasarkan HbA1c", 

    description="Silakan isi data medis pasien di bawah ini, lalu klik Submit untuk melihat rekomendasi penanganan otomatis dari model AI." 

) 
demo.launch(share=True) 

 
