import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
from streamlit_option_menu import option_menu

LPDP = Image.open("brin.png")#https://lpdp.kemenkeu.go.id/
#BRIN = "brin.png"
st.logo(LPDP,link="https://lpdp.kemenkeu.go.id/",icon_image="None")
selected = option_menu(
    menu_title=None,
    options = ["Beranda", "Prediksi", "Kontak", "Lokasi"],
    icons = ["house","gear","envelope","pin"],
    menu_icon = "cast",
    default_index = 0,
    orientation = "horizontal",
)
if selected == "Beranda":
    st.title("Prediksi Performa Mesin Diesel Menggunakan Algoritma Pembelajaran Mesin (Machine Learning)")
    st.write(
    "Selamat datang pada situs web ini. Situs web ini dibuat untuk melakukan prediksi performa mesin diesel menggunakan pembelajaran mesin (machine learning).")
    st.write(
    "Dataset yang digunakan untuk membangun model pembelajaran mesin ini diperoleh dari [1]. Mesin yang digunakan adalah mesin diesel 4 langkah dengan merk Yanmar dengan kapasitas 7.5 kW dan bahan bakar diesel/biodiesel. Mesin tersebut digunakan sebagai sumber energi listrik (genset) dan diberikan beban listrik")
    st.write(
    "Masukan data (input) yang digunakan oleh model pembelajaran mesin ini adalah kecepatan (0 - 1200 rpm), beban (0 - 4000 Watt), persentase biodiesel (0 - 50%), dan suhu campuran biodiesel (26 - 60 oC).")
    st.write(
    "Algoritma ML yang digunakan dalam laman web ini adalah Artificial Neural Network dengan Multilayer Perceptron (MLPNN) untuk torsi dan XGBoost untuk efisiensi termal dan SFC.")
    st.write(
    "Luaran data (output) yang dihasilkan oleh model pembelajaran mesin ini adalah torsi (Nm), specific fuel consumption (g/kWh), dan efisiensi termal (%).")
if selected == "Prediksi":
    st.title("Prediksi Pembelajaran Mesin")
    st.markdown("Masukan nilai yang digunakan untuk memprediksi performa mesin diesel")
    st.header("Input Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        load = st.slider("Load (Watt)", 1.0, 4000.0, 1000.0)
        speed = st.slider("Speed (rpm)", 1.0, 1200.0, 800.0)
    with col2:
        bio_d = st.slider("Persentasi Biodiesel (%)", 1.0, 50.0, 0.0)
        bio_bt = st.slider("Temperatur Campuran (deg C)", 0.1, 60.0, 0.0)
    if st.button("Prediksi!"):
        result = predict(np.array([load, speed, bio_d, bio_bt]))
        st.text(result[0])
if selected == "Kontak":
    st.title("Tim Riset Inovasi Indonesia Maju - Lembaga Pengelola Dana Pendidikan (RIIM-LPDP) ")
    st.write("Kelompok Riset Pemodelan Sarana Transportasi Berkelanjutan")
    st.markdown("- Nilam Sari Octaviani")
    st.markdown("- Rizqon Fajar")
    st.markdown("- Kurnia Fajar Adhi Sukra")
    st.markdown("- Sigit Tri Atmaja")
    st.markdown("- Fitra Hidiyanto")
    st.markdown("- Raditya Hendra Pratama")
    st.markdown("- Dhani Avianto Sugeng")
    st.markdown("- Ardani Cesario Zuhri")
    st.write("Kelompok Riset Bioenergi dan Energi Alternatif")
    st.markdown("- Arya Bhaskara Adiprabowo")
if selected == "Lokasi":
    puspiptek = pd.DataFrame(({'lat': [-6.3476678],'lon': [-106.66186]}))
    st.map(puspiptek)#-6.3476678,106.66186
# Create a DataFrame with latitude and longitude data