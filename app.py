import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Rekomendasi Wisata Pesisir Barat", page_icon="🌊", layout="wide")

# --- DATASET WISATA PESISIR BARAT ---
@st.cache_data
def load_data():
    data = [
        {"kecamatan": "Bangkunat", "nama": "Pantai Ujung Belimbing", "deskripsi": "Pantai alami eksotis di ujung semenanjung selatan."},
        {"kecamatan": "Bangkunat", "nama": "Pantai Teluk Bengkunat", "deskripsi": "Area pantai yang bersisian dengan kawasan pelabuhan nelayan."},
        {"kecamatan": "Bangkunat", "nama": "Pantai Pelabuhan Nusantara", "deskripsi": "Pesisir pantai strategis yang menghadap laut lepas."},
        {"kecamatan": "Bangkunat", "nama": "Pantai Curup Indah", "deskripsi": "Terletak di kawasan Pekon Kota Jawa."},
        {"kecamatan": "Bangkunat", "nama": "Pantai KEM", "deskripsi": "Kawasan pantai resort yang asri dan tenang."},
        {"kecamatan": "Bangkunat", "nama": "Sungai Way Cangkuk", "deskripsi": "Aliran sungai jernih di dalam hutan tropis untuk susur sungai."},
        {"kecamatan": "Bangkunat", "nama": "Wisata Alam Tampang Belimbing", "deskripsi": "Petualangan rimba pesisir terisolasi di dalam hutan TNBBS."},
        {"kecamatan": "Ngaras", "nama": "Pantai Siging", "deskripsi": "Pantai historis yang memiliki bekas dermaga peninggalan era kolonial Belanda."},
        {"kecamatan": "Ngaras", "nama": "Pelabuhan Nelayan Siging", "deskripsi": "Pusat pantai pendaratan kapal-kapal nelayan tradisional."},
        {"kecamatan": "Ngaras", "nama": "Pantai Suka Negara", "deskripsi": "Kawasan pantai pasir putih alami di pekon Sukanegara."},
        {"kecamatan": "Ngaras", "nama": "Pantai Hujung Langgar", "deskripsi": "Pantai tersembunyi dengan batuan karang indah di Pekon Baturaja."},
        {"kecamatan": "Ngaras", "nama": "Rhino Camp", "deskripsi": "Pos ekowisata dan petualangan alam di kawasan hutan penyangga TNBBS."},
        {"kecamatan": "Ngambur", "nama": "Pantai Labuhan Bakhu", "deskripsi": "Pantai landai berpagar karang alami pemecah ombak."},
        {"kecamatan": "Ngambur", "nama": "Pantai Sumber Agung", "deskripsi": "Pantai asri berpasir hitam-abu yang luas."},
        {"kecamatan": "Ngambur", "nama": "Pantai Way Batang", "deskripsi": "Kawasan pesisir pekon Way Batang yang tenang."},
        {"kecamatan": "Ngambur", "nama": "Air Terjun Simpang Luh", "deskripsi": "Destinasi air terjun bertingkat yang dikelilingi hutan perkebunan warga."},
        {"kecamatan": "Pesisir Selatan", "nama": "Pantai Tanjung Setia", "deskripsi": "Destinasi ikonik kelas dunia bagi para peselancar mancanegara."},
        {"kecamatan": "Pesisir Selatan", "nama": "Pantai Karang Nyimbor", "deskripsi": "Spot selancar dengan ombak tipe left-hander yang panjang."},
        {"kecamatan": "Pesisir Selatan", "nama": "Pantai Marang", "deskripsi": "Pantai indah yang terletak di pesisir Pekon Marang."},
        {"kecamatan": "Pesisir Selatan", "nama": "Pantai Melasti", "deskripsi": "Pantai sakral yang digunakan umat Hindu untuk ritual upacara keagamaan."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Mandiri", "deskripsi": "Garis pantai landai tanpa karang sepanjang berkilo-kilometer."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Walur", "deskripsi": "Pantai berair tenang dengan lanskap rumput hijau di sekelilingnya."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Wisata Ilahan", "deskripsi": "Spot pantai tersembunyi di dalam area Pekon Walur."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Lintik", "deskripsi": "Pantai indah yang lokasinya tepat berada di tepi Jalan Lintas Barat."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Ringis", "deskripsi": "Pantai karang kecil yang sering dijadikan tempat memancing warga."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Benawang", "deskripsi": "Area pantai alami tersembunyi di selatan Krui."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Cemara", "deskripsi": "Pantai rindang yang dikelilingi oleh jajaran pohon cemara laut."},
        {"kecamatan": "Krui Selatan", "nama": "Pantai Banana", "deskripsi": "Pantai berpasir bersih di kawasan Pekon Bangun Negara."},
        {"kecamatan": "Krui Selatan", "nama": "Air Terjun Way Nyarecik", "deskripsi": "Wisata air terjun alami di kawasan Pekon Paku Negara."},
        {"kecamatan": "Krui Selatan", "nama": "Air Terjun Way Basohan", "deskripsi": "Aliran air terjun jernih di dalam kawasan hutan pegunungan."},
        {"kecamatan": "Pesisir Tengah", "nama": "Pantai Labuhan Jukung", "deskripsi": "Ikon wisata utama di pusat ibu kota (Krui) dengan fasilitas terlengkap."},
        {"kecamatan": "Pesisir Tengah", "nama": "Pantai Krui", "deskripsi": "Tepian pantai pasir putih yang membentang di sekitar area pusat kota."},
        {"kecamatan": "Karya Penggawa", "nama": "Pantai Penggawa Lima Tengah", "deskripsi": "Area pesisir berbatu karang yang indah."},
        {"kecamatan": "Karya Penggawa", "nama": "Gua Matu", "deskripsi": "Situs wisata alam gua tebing pantai bersejarah yang dihuni ribuan kelelawar."},
        {"kecamatan": "Pesisir Utara", "nama": "Pantai Batu Tihang", "deskripsi": "Pantai ikonik dengan sebuah batu karang raksasa tegak menyerupai tiang."},
        {"kecamatan": "Pesisir Utara", "nama": "Pantai Sharling", "deskripsi": "Objek wisata pantai yang baru dikembangkan di kawasan Pekon Way Nukak."},
        {"kecamatan": "Lemong", "nama": "Pantai Tanjung Jati", "deskripsi": "Pantai karang berpasir putih di dekat perbatasan provinsi."},
        {"kecamatan": "Lemong", "nama": "Pantai Batu Mirau", "deskripsi": "Ikon pantai dengan dua pilar batu karang eksotis di tepian air."},
        {"kecamatan": "Lemong", "nama": "Pantai Parda Haga", "deskripsi": "Pantai asri berombak besar di Pekon Parda Haga."},
        {"kecamatan": "Lemong", "nama": "Pantai Bambang", "deskripsi": "Pantai alami berkarang dengan latar perbukitan hijau."},
        {"kecamatan": "Lemong", "nama": "Pantai Pekon Balak", "deskripsi": "Wilayah pesisir utara berair jernih di dekat perbatasan Bengkulu."},
        {"kecamatan": "Pulau Pisang", "nama": "Pantai Pasir Putih Pulau Pisang", "deskripsi": "Pantai pasir putih bersih yang mengelilingi seluruh pulau utama."},
        {"kecamatan": "Pulau Pisang", "nama": "Perairan Lumba-Lumba (Dolphin Spotting)", "deskripsi": "Wisata alam laut lepas untuk melihat kawanan lumba-lumba liar."}
    ]
    return pd.DataFrame(data)

df = load_data()

# --- LOGIKA CONTENT-BASED FILTERING ---
df['konten'] = df['kecamatan'] + " " + df['deskripsi']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['konten'])

def get_recommendation(user_input, top_n=5):
    user_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df_result = df.copy()
    df_result['skor_kemiripan'] = similarity
    df_result = df_result.sort_values(by='skor_kemiripan', ascending=False)
    df_result = df_result[df_result['skor_kemiripan'] > 0].head(top_n)
    return df_result

# --- ANTARMUKA WEB ---
st.title("🏄‍♂️ Sistem Rekomendasi Wisata Pesisir Barat")
st.markdown("Aplikasi Skripsi: Analisis Sistem Rekomendasi Wisata Menggunakan Metode **Content-Based Filtering**.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔍 Cari Wisata")
    user_query = st.text_area("Ketik kriteria liburan Anda:", placeholder="Contoh: wisata alam hutan, pantai pasir putih...")
    jumlah_rek = st.slider("Jumlah Rekomendasi:", 1, 10, 5)
    btn_cari = st.button("Cari Rekomendasi", type="primary", use_container_width=True)

with col2:
    st.header("🎯 Hasil Rekomendasi")
    if btn_cari:
        if user_query.strip() == "":
            st.warning("Mohon isi kotak pencarian terlebih dahulu!")
        else:
            hasil = get_recommendation(user_query, top_n=jumlah_rek)
            if hasil.empty:
                st.error("Maaf, tidak ada wisata yang cocok.")
            else:
                st.success(f"Ditemukan {len(hasil)} rekomendasi yang cocok!")
                for index, row in hasil.iterrows():
                    persen = round(row['skor_kemiripan'] * 100, 1)
                    with st.container():
                        st.subheader(f"📍 {row['nama']} ({persen}% Cocok)")
                        st.caption(f"Kecamatan: {row['kecamatan']}")
                        st.write(f"**Info:** {row['deskripsi']}")
                        st.markdown("---")
