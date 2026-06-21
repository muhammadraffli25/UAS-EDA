import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==========================================
# KONFIGURASI HALAMAN & JUDUL
# ==========================================
st.set_page_config(page_title="Dashboard UAS EDA", layout="wide")

st.title("LAPORAN UAS: EXPLORATORY DATA ANALYSIS & SENTIMENT ANALYSIS")
st.markdown("**Nama:** Muhammad Raffli")
st.markdown("**NIM:** 2025492005")
st.markdown("**Topik:** Analisis Sentimen Ulasan Aplikasi Zoom Cloud Meetings pada Google Play Store")
st.markdown("**Link Dashboard:** [Klik di sini untuk membuka web](MASUKKAN_LINK_LU_DI_SINI)")
st.markdown("---")

# ==========================================
# LOAD DATA FINAL
# ==========================================
try:
    df_sentimen = pd.read_csv('data_sentimen.csv')
except FileNotFoundError:
    st.error("⚠️ File 'data_sentimen.csv' tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py!")
    st.stop()


# ==========================================
# BAGIAN A: SCRAPING & TABEL MENTAH
# ==========================================
st.header("A. Pengambilan Data Ulasan (Scraping)")
st.markdown("""
**Narasi Pengambilan Data:**
Data yang digunakan dalam penelitian ini diperoleh melalui proses *scraping* ulasan pengguna aplikasi Zoom Cloud Meetings langsung dari platform Google Play Store dengan total sampel sebanyak 1.000 data. Data mentah ini ditarik untuk mendapatkan gambaran asli mengenai pendapat pengguna sebelum masuk ke tahap pemrosesan teks.
""")

st.markdown("**Tabel Sampel Data Hasil Scraping (Preview 99 Data Teratas):**")
# Menampilkan kolom mentah/keseluruhan sebagai representasi hasil scraping awal
st.dataframe(df_sentimen.head(99), height=250)
st.markdown("---")


# ==========================================
# BAGIAN B: REASON TOPIK
# ==========================================
st.header("B. Alasan Pemilihan Topik")
st.markdown("""
**Alasan Pemilihan Topik (Zoom Cloud Meetings):**
Saya memilih Zoom karena berdasarkan pengalaman saya pribadi, ini adalah aplikasi komunikasi yang sangat penting dan dibutuhkan dengan karakteristik data sebagai berikut:
* **Skala Besar:** Aplikasi ini digunakan oleh masyarakat yang sangat banyak secara global dari berbagai macam latar belakang.
* **Beragam:** Karena penggunanya luas dan selalu dibutuhkan, maka keluhan, kritik fitur, hingga pujian di kolom ulasan menjadi sangat beragam.
* **Relevan:** Memproses data Zoom sangat sepadan karena hasilnya memberikan gambaran nyata mengenai kepuasan masyarakat terhadap teknologi yang dipakai setiap hari.
""")
st.markdown("---")


# ==========================================
# BAGIAN C: PEMBERSIHAN DATA & TABEL BERSIH
# ==========================================
st.header("C. Pembersihan Data Teks (Preprocessing)")
st.markdown("""
**Narasi Metode Pembersihan Data:**
Teks ulasan mentah hasil *scraping* masih mengandung banyak karakter tidak penting. Oleh karena itu, dilakukan tahapan pembersihan data yang meliputi:
1. **Cleansing:** Menghilangkan karakter pengganggu seperti tautan (URL), *mention* (@), *hashtag* (#), angka, tanda baca, dan simbol emotikon.
2. **Case Folding:** Menyeragamkan semua huruf teks menjadi huruf kecil.
3. **Tokenisasi & Stopwords Removal:** Memotong kalimat menjadi kata tunggal dan membuang kata hubung (seperti "yang", "di", "dan") yang tidak memiliki makna sentimen.
4. **Stemming:** Mengembalikan kata berimbuhan ke kata dasarnya (misal: "membantu" menjadi "bantu").
""")

st.markdown("**Tabel Sampel Hasil Pembersihan Data (Preview 99 Data Teratas):**")
# Menampilkan kolom teks yang sudah bersih sebagai pembuktian proses preprocessing
if 'teks_bersih' in df_sentimen.columns:
    st.dataframe(df_sentimen[['teks_bersih']].head(99), height=250)
else:
    st.dataframe(df_sentimen.head(99), height=250)
st.markdown("---")


# ==========================================
# BAGIAN D: WORDCLOUD & ANALISIS KATA
# ==========================================
st.header("D. Visualisasi Kata Kunci (WordCloud)")

# Proses membuat WordCloud dari kolom teks_bersih
teks_all = " ".join(str(teks) for teks in df_sentimen['teks_bersih'].dropna())

fig1, ax1 = plt.subplots(figsize=(12, 4))
wordcloud = WordCloud(width=800, height=300, background_color='white', colormap='viridis').generate(teks_all)
ax1.imshow(wordcloud, interpolation='bilinear')
ax1.axis('off')
st.pyplot(fig1)

st.markdown("""
**Analisis WordCloud (Pengelompokan Kata):**
Visualisasi WordCloud di atas membedah fokus pembicaraan pengguna ke dalam dua kelompok utama:

1. **Berdasarkan Sentimen (Baik vs Buruk):**
   * **Kata Indikator Baik (Pujian):** Didominasi oleh kata *"bantu"*, *"bagus"*, *"mantap"*, *"lancar"*, dan *"mudah"*. Ini menunjukkan tingginya kepuasan pengguna terhadap kelancaran aplikasi.
   * **Kata Indikator Buruk (Keluhan):** Terdapat kata yang cukup besar seperti *"tidak"*, *"susah"*, *"bug"*, *"gagal"*, dan *"gabisa"*. Kata *"tidak"* yang dominan menunjukkan banyaknya kalimat keluhan pengguna.
2. **Berdasarkan Jenis Kata (Objek vs Kata Sifat/Kerja):**
   * **Kata Objek Fitur:** Menyoroti bagian dalam aplikasi seperti *"kamera"*, *"background"*, *"fitur"*, *"video"*, dan *"suara"*.
   * **Kata Pengalaman Pengguna:** Menjelaskan tujuan penggunaan seperti *"belajar"*, *"meeting"*, *"update"*, dan *"login"*.
   * **Kesimpulan Analisis:** Keluhan dan kepuasan terbesar pengguna Zoom berpusat pada **kestabilan kamera dan suara** saat digunakan untuk **belajar atau meeting**, terutama saat memakai fitur **background** atau setelah melakukan **update** aplikasi.
""")
st.markdown("---")


# ==========================================
# BAGIAN E: ANALISIS SENTIMEN & DISTRIBUSI
# ==========================================
st.header("E. Analisis Sentimen & Distribusi Hasil")

# Membagi layar jadi 2 kolom untuk efisiensi visual dashboard
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("**Tabel Sampel Hasil Klasifikasi Sentimen (Preview 99 Data Teratas):**")
    st.dataframe(df_sentimen[['teks_bersih', 'skor_sentimen', 'kategori_sentimen', 'rating']].head(99), height=350)

with col2:
    st.markdown("**Distribusi Total Sentimen 1.000 Data (Pie Chart):**")
    sentimen_counts = df_sentimen['kategori_sentimen'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    
    # Warna: Hijau untuk Positif, Merah untuk Negatif
    warna = ['#4CAF50' if x == 'Positif' else '#F44336' for x in sentimen_counts.index]
    
    ax2.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90, colors=warna)
    ax2.axis('equal')
    st.pyplot(fig2)

st.markdown("""
**Analisis Hasil Sentimen:**

* **1. Analisis Anomali Tabel (Keterbatasan Kamus Sistem):**
  Berdasarkan pengamatan pada tabel, ditemukan hal menarik di mana ulasan bernada sangat baik dan memiliki rating bintang 5 justru banyak mendapat **skor 0.3** dan **0.0**.
  * **Dominasi Skor 0.3:** Terjadi karena kata positif yang digunakan pengguna tidak terbaca nilai maksimalnya oleh kamus sistem (terutama jika ulasan menggunakan bahasa gaul atau singkatan lokal).
  * **Dominasi Skor 0.0:** Terjadi karena ulasan terlalu singkat (hanya mengetik "ok", "sip", atau mengirim emotikon). Sistem gagal mendeteksi sentimen sehingga memberi nilai netral (0.0). Namun, penentuan hasil akhir (Positif/Negatif) tetap akurat karena dibantu oleh nilai rating bintang yang diberikan pengguna.

* **2. Analisis Distribusi Grafik (Pie Chart):**
  * **Mayoritas (67.6% Positif):** Menegaskan bahwa performa aplikasi Zoom secara keseluruhan sangat memuaskan penggunanya, sejalan dengan banyaknya kata *"bantu"* dan *"bagus"* pada WordCloud.
  * **Minoritas (32.4% Negatif):** Angka ini adalah bagian keluhan teknis pengguna yang nyata. Ini mewakili keluhan pada kata *"susah"*, *"bug"*, atau *"gagal"* di WordCloud yang umumnya terjadi saat proses *login* atau masalah jaringan pada perangkat tertentu.
""")