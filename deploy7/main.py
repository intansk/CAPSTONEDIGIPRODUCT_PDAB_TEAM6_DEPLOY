import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
import pickle
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write("""
TEAM 6
1. INTAN MELANI SUKMA (2209116028)
2. SILVA JEN RETNO (2209116019)
3. FINA ANRIANI (2209116051)
""")

# Sidebar untuk navigasi
with st.sidebar:
    selected = option_menu('Dashboard',
                           ['Home',
                            'Data Visualization',
                            'Columns',
                            'Classification'],

                            icons = ['house-fill', 
                                     'image-fill',
                                     'layout-three-columns',
                                     'arrows-angle-contract'],
                            default_index = 0)


# Home page
if selected == 'Home':

    #page tittle
    # Menampilkan gambar dari file lokal
    from PIL import Image
    image = Image.open('dataset-cover (1).jpg')

    # Menampilkan gambar dengan ukuran yang disesuaikan
    st.image(image, caption='', use_column_width=True)

    st.subheader('Analisis Data Starbucks Memprediksi Retensi Pelanggan (Starbucks Customers Survey)')
    st.write('''Dataset Analisis Data Starbucks untuk Memprediksi Retensi Pelanggan, juga dikenal sebagai Starbucks Customers Survey, merupakan kumpulan data yang disusun untuk menganalisis perilaku pelanggan Starbucks dengan tujuan memprediksi faktor-faktor yang memengaruhi retensi pelanggan. Dataset ini umumnya terdiri dari berbagai variabel yang mencakup informasi seperti karakteristik demografis pelanggan (misalnya usia, jenis kelamin, lokasi), perilaku pembelian (misalnya frekuensi kunjungan, jumlah belanja, produk yang dibeli), preferensi produk, serta tanggapan dari survei kepuasan pelanggan.
    ''')
    st.write('''Tujuan dari analisis data ini adalah untuk mengidentifikasi pola-pola dan tren yang terkait dengan retensi pelanggan di Starbucks. Dengan memahami faktor-faktor yang mempengaruhi retensi pelanggan, perusahaan dapat mengambil tindakan yang sesuai untuk meningkatkan pengalaman pelanggan dan mempertahankan basis pelanggannya.''')
    st.write('''Dataset ini dapat menjadi sumber daya yang berharga bagi para peneliti, analis bisnis, dan praktisi pemasaran untuk mengembangkan strategi yang lebih efektif dalam mempertahankan pelanggan serta meningkatkan kepuasan dan loyalitas pelanggan di rantai kopi Starbucks.''')
    st.write('Berikut ini merupakan link dari dataset yang digunakan: https://www.kaggle.com/datasets/mahirahmzh/starbucks-customer-retention-malaysia-survey')
    
    # Read data
    st.write('DATASET AWAL')
    df = pd.read_csv('Starbucks customer survey.csv')
    st.write (df)
    st.write('Dataset awal ini adalah data mentah yang diperoleh langsung dari sumbernya tanpa melalui proses apapun. Dataset ini memiliki struktur, format, dan kualitas yang bervariasi tergantung dari sumber data aslinya.')
    st.write('Dataset ini biasanya memerlukan pembersihan dan transformasi lebih lanjut sebelum dapat digunakan untuk analisis atau aplikasi lainnya.')
    
    st.write('DATASET AKHIR')
    df = pd.read_csv('Data Cleaned.csv')
    st.write (df)
    st.write('Dataset akhir ini adalah data yang telah melalui proses pembersihan (cleaned) dan transformasi untuk memastikan kualitas, konsistensi, dan integritas data yang lebih baik.')
    st.write('Dataset akhir ini memiliki struktur dan format yang konsisten, data yang lengkap dan tidak ada duplikat, serta siap untuk digunakan dalam analisis atau aplikasi lainnya. Dataset ini memudahkan analisis dan interpretasi data, serta menghasilkan insight dan hasil yang lebih akurat dan relevan.')

if selected == 'Data Visualization':
    df = pd.read_csv('Starbucks customer survey.csv')
    # Visualisasi dengan Streamlit
    
    def enter1():
        st.write("<br>", unsafe_allow_html=True)

    def enter2():
        st.write("<br><br>", unsafe_allow_html=True)

    def main_title():
        st.title("❝Analisis Data Starbucks untuk Memprediksi Retensi Pelanggan (Starbucks Customers Survey)❞")

    # data awal
    def data():
        return pd.read_csv('Starbucks customer survey.csv')

    # data clean
    def data2():
        return pd.read_csv('Data Cleaned.csv')

    def visualisasi1(df):
        df_temp = df.copy()
        df_temp['age'] = df_temp['age'].replace({0: "<20", 1: "20-29", 2: "30-39", 3: ">40"})
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df_temp, x='age', bins=4, kde=False, color='pink', ax=ax)
        ax.set_title('Distribusi Umur')
        ax.set_xlabel('Umur')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)

    def visualisasi2(df):
        df_temp = df.copy()
        df_temp['gender'] = df_temp['gender'].replace({0: 'Male', 1: 'Female'})
        df_temp['visitNo'] = df_temp['visitNo'].replace({0: 'Daily', 1: 'Weekly', 2: 'Monthly', 3: 'Never'})
        visitNo_gender_counts = df_temp.groupby(['visitNo', 'gender']).size().unstack(fill_value=0)
        visitNo_gender_counts.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
        plt.title('Perbandingan Frekuensi Kunjungan Berdasarkan Gender')
        plt.xlabel('Frekuensi Kunjungan')
        plt.ylabel('Jumlah')
        plt.xticks(rotation=45)
        plt.legend(title='Gender')
        st.pyplot() 

    def visualisasi3(df):
        df_temp = df.copy()
        df_temp['loyal'] = df_temp['loyal'].replace({0: 'Loyal', 1: 'Tidak Loyal'})
        loyal_counts = df_temp['loyal'].value_counts()

        # Mengatur warna yang diinginkan
        custom_colors = ['green', 'lightgrey']

        plt.figure(figsize=(6, 6))
        plt.pie(loyal_counts, labels=loyal_counts.index, autopct='%1.1f%%', startangle=90, colors=custom_colors)
        plt.title('Komposisi Loyalitas Pelanggan')
        plt.axis('equal')
        st.pyplot()

    def visualisasi4(df):
        # Relationship (Hubungan)
        df_temp = df.copy()
        df_temp['gender'] = df_temp['gender'].replace({0: 'Male', 1: 'Female'})
        df_temp['visitNo'] = df_temp['visitNo'].replace({0: 'Daily', 1: 'Weekly', 2: 'Monthly', 3: 'Never'})
        df_temp['spendPurchase'] = df_temp['spendPurchase'].replace({0: 'Zero', 1: '<RM20', 2: 'RM20 - RM40', 3: '>RM40'})
        df_temp['loyal'] = df_temp['loyal'].replace({0: 'Loyal', 1: 'Tidak Loyal'})
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='loyal',
            y='spendPurchase',
            hue='gender',
            palette='Set3',
            data=df_temp,
            ax=ax)
        ax.set_title('Hubungan Loyalitas Pelanggan dengan Total Pengeluaran')
        ax.set_xlabel('Loyalitas Pelanggan')
        ax.set_ylabel('Total Pengeluaran (RM)')
        ax.set_xticks([0, 1])  # Label untuk sumbu x
        ax.set_xticklabels(['Loyal', 'Tidak Loyal'])  
        ax.set_yticks([0, 1, 2, 3])  # Label untuk sumbu y
        ax.set_yticklabels(['Zero', '<RM20', 'RM20 - RM40', '>RM40'])  
        st.pyplot(fig)

    def visualisasi5(df):
        # Composition (Komposisi)
        # lebih banyak pelanggan perempuan atau laki-laki?
        df_temp = df.copy()
        df_temp['gender'] = df_temp['gender'].replace({0: "male", 1: "female"})
        Transportation_counts = df_temp['gender'].value_counts()

        colors = ['skyblue', 'gray']
        fig, ax = plt.subplots()
        Transportation_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=colors, ax=ax)
        ax.set_ylabel('')
        plt.title('Komposisi Pelanggan Berdasarkan Gender')
        plt.axis('equal')
        st.pyplot(fig)

    def visualisasi6(df):
        # Composition (Komposisi)
        # dari keseluruhan pelanggan, lebih banyak dari kalangan apa?

        status_labels = {
            0: 'Student',
            1: 'Self-Employed',
            2: 'Employed',
            3: 'Housewife'
        }
        status_counts = df['status'].value_counts()

        # Buat pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(status_counts, labels=[status_labels[i] for i in status_counts.index], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'pink'])
        ax.set_title('Pie Chart Status Pelanggan', fontsize=16)
        ax.axis('equal')  # Agar pie chart menjadi lingkaran
        ax.legend(title='Status', loc='upper right')
        st.pyplot(fig)  # Menggunakan st.pyplot() untuk menampilkan plot di Streamlit

    def visualisasi7(df):
        # Composition (Komposisi)
        # Seberapa sering pelanggan starbucks mengunjungi starbucks?

        visit_labels = {
            0: 'Daily',
            1: 'Weekly',
            2: 'Monthly',
            3: 'Never'}

        visit_counts = df['visitNo'].value_counts()

        # Buat pie chart dengan label yang sudah diganti
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(visit_counts, labels=[visit_labels[i] for i in visit_counts.index], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax.set_title('Pie Chart Frekuensi Kunjungan', fontsize=16, color='navy')
        ax.axis('equal')
        ax.legend(title='Frekuensi', loc='upper right')
        st.pyplot(fig)

    def visualisasi8(df):
        # Composition (Komposisi)
        # Berapa banyak pengeluaran pelanggan ketika mengunjungi starbucks?

        spend_labels = {
            0: 'Zero',
            1: 'Less than RM20',
            2: 'RM20 to RM40',
            3: 'More than RM40'
        }
        spend_counts = df['spendPurchase'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(spend_counts, labels=[spend_labels[i] for i in spend_counts.index], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax.set_title('Pie Chart Pengeluaran Pelanggan', fontsize=16, color='navy')
        ax.axis('equal')
        ax.legend(title='Pengeluaran', loc='upper right')
        st.pyplot(fig)

    def visualisasi9(df):
        df_temp = df.copy()
        df_temp['age'] = df_temp['age'].replace({0: "<20", 1: "20-29", 2: "30-39", 3: ">40"})
        df_temp['loyal'] = df_temp['loyal'].replace({0: 'Loyal', 1: 'Tidak Loyal'})
        
        # Mengatur palet warna yang diinginkan
        custom_palette = {'Loyal': 'green', 'Tidak Loyal': 'lightgrey'}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='age', hue='loyal', palette=custom_palette, data=df_temp, ax=ax)
        ax.set_title('Perbandingan Loyalitas Berdasarkan Usia')
        ax.set_xlabel('Usia')
        ax.set_ylabel('Jumlah')
        
        st.pyplot(fig)

    def visualisasi10(df):
        # Comparison (Perbandingan)

        df_new = df.copy()
        df_new['gender'] = df_new['gender'].replace({0: 'male', 1: 'female'})
        df_new['loyal'] = df_new['loyal'].replace({0: 'iya', 1: 'tidak'})
        
        # Mengatur palet warna yang diinginkan
        custom_palette = {'iya': 'green', 'tidak': 'lightgrey'}

        fig, ax = plt.subplots()
        sns.countplot(x="gender",
                    hue="loyal",
                    palette=custom_palette,
                    data=df_new,
                    order=['male', 'female'],
                    hue_order=['iya', 'tidak'])

        plt.title('Loyalty by gender')
        plt.xlabel('gender')
        plt.ylabel('Count')
        plt.legend(title='Loyal', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

    def visualisasi11(df):
        # Comparison (Perbandingan)
        plt.figure(figsize=(10, 6))
        sns.countplot(x='age', hue='loyal', palette='vlag', data=df)
        plt.title('Perbandingan Loyalitas Berdasarkan Usia')
        plt.xlabel('Usia')
        plt.ylabel('Jumlah')
        plt.legend(title='Loyal')
        st.pyplot()

    def visualisasi12(df):
        # Relationship (Hubungan)
        # apakah pelanggan yang mempunyai membership sudah pasti loyal?
        df_new = df.copy()
        df_new['membershipCard'] = df_new['membershipCard'].replace({0: 'punya', 1: 'tidak'})
        df_new['loyal'] = df_new['loyal'].replace({0: 'iya', 1: 'tidak'})

        # Mengatur palet warna yang diinginkan
        custom_palette = {'iya': 'green', 'tidak': 'lightgrey'}

        fig, ax = plt.subplots()
        sns.countplot(x="membershipCard",
                    hue="loyal",
                    palette=custom_palette,
                    data=df_new,
                    order=['punya', 'tidak'],
                    hue_order=['iya', 'tidak'])

        plt.title('Count of Membership Card by Loyal')
        plt.xlabel('Membership Card')
        plt.ylabel('Count')
        plt.legend(title='Loyal', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

    def visualisasi13(df):
        # Composition (Komposisi)
        plt.figure(figsize=(8, 8))
        df['membershipCard'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'grey'])
        plt.title('Komposisi Kepemilikan Kartu Keanggotaan')
        plt.ylabel('')
        st.pyplot()

    def relationship(df2):
        df2_corr = df2.corr(numeric_only=True)
        fig = px.imshow(df2_corr)
        st.plotly_chart(fig)

    df = data()
    df2 = data()
    def main():
        main_title()
        st.subheader('Visualization')
        st.write("Berikut merupakan beberapa hasil visualisasi dari dataset Starbucks Customers Survey >>")

        st.header('COMPARISON')
        st.write('Visualisasi 2')
        visualisasi2(df)
        st.write('Grafik ini menunjukkan bahwa perempuan lebih sering mengunjungi daripada laki-laki. Perempuan paling sering mengunjungi secara bulanan, sedangkan laki-laki paling sering tidak pernah mengunjungi.')
        st.write('========================================================================================')

        st.write('visualisasi 9')
        visualisasi9(df)
        st.write('Grafik ini menunjukkan bahwa loyalitas pelanggan berkaitan dengan usia dan total pengeluaran. Pelanggan loyal dengan usia 30-39 tahun dan total pengeluaran lebih dari RM40 adalah yang paling banyak.')
        st.write('========================================================================================')

        st.write('visualisasi 10')
        visualisasi10(df)
        st.write('Berdasarkan visualisasi tersebut, dapat disimpulkan bahwa perempuan lebih loyal dibandingkan dengan laki-laki dalam menjadi pelanggan Starbucks.')
        st.write('========================================================================================')

        st.write('visualisasi 12')
        visualisasi12(df)
        st.write('Visualisasi ini menunjukkan bahwa memiliki kartu membership berhubungan dengan loyalitas pelanggan yang lebih tinggi.')
        st.write('========================================================================================')

        st.header('COMPOSITION')
        st.write('visualisasi 3')
        visualisasi3(df)
        st.write('Mayoritas pelanggan loyal kepada perusahaan. Hal ini ditunjukkan dengan persentase pelanggan loyal yang lebih tinggi dibandingkan persentase pelanggan tidak loyal. Hal ini menunjukkan bahwa pelanggan puas dengan produk atau layanan yang ditawarkan perusahaan dan bersedia untuk terus menjalin hubungan dengan perusahaan. Meskipun demikian, masih ada 17.4% pelanggan yang tidak loyal. Hal ini menunjukkan bahwa masih ada ruang bagi perusahaan untuk meningkatkan kepuasan pelanggan dan mencegah pelanggan berpindah ke perusahaan lain.')
        st.write('========================================================================================')

        st.write('visualisasi 5')
        visualisasi5(df)
        st.write('Berdasarkan visualisasi diatas, terdapat 52,2% pelanggan perempuan dan 47,8% pelanggan laki-laki. Hal ini menunjukkan bahwa jumlah pelanggan perempuan lebih banyak daripada laki-laki. Persentase pelanggan perempuan: Warna biru muda pada diagram lingkaran mewakili persentase pelanggan perempuan, yaitu 52,2%. Hal ini menunjukkan bahwa lebih dari setengah pelanggan adalah perempuan. Persentase pelanggan laki-laki: Warna abu-abu pada diagram lingkaran mewakili persentase pelanggan laki-laki, yaitu 47,8%. Hal ini menunjukkan bahwa hampir setengah dari pelanggan adalah laki-laki. Sehingga dapat disimpulkan bahwa terdapat lebih banyak pelanggan perempuan daripada laki-laki.')
        st.write('========================================================================================')

        st.write('visualisasi 6')
        visualisasi6(df)
        st.write('Berdasarkan visualisasi tersebut, dapat disimpulkan bahwa pelanggan Starbucks didominasi oleh karyawan dan pelajar. Hal ini menunjukkan bahwa Starbucks merupakan tempat yang populer bagi orang-orang yang ingin bekerja atau belajar dengan suasana yang nyaman dan santai.')
        st.write('========================================================================================')

        st.write('visualisasi 7')
        visualisasi7(df)
        st.write('Sekali Sehari (Daily): Kelompok ini merupakan kelompok terkecil dengan persentase 1.8%. Hal ini menunjukkan bahwa hanya sebagian kecil pelanggan Starbucks yang mengunjungi Starbucks setiap hari. Sekali Seminggu (Weekly): Kelompok ini memiliki persentase 8.0%. Hal ini menunjukkan bahwa sebagian besar pelanggan Starbucks mengunjungi Starbucks setidaknya sekali seminggu. Sekali Sebulan (Monthly): Kelompok ini menempati urutan kedua dengan persentase 23.0%. Hal ini menunjukkan bahwa banyak pelanggan Starbucks yang mengunjungi Starbucks setidaknya sekali sebulan. Jarang/Tidak Pernah (Never): Kelompok ini merupakan kelompok terbesar dengan persentase 67.3%. Hal ini menunjukkan bahwa sebagian besar pelanggan Starbucks jarang atau tidak pernah mengunjungi Starbucks. Berdasarkan pie chart tersebut, dapat disimpulkan bahwa sebagian besar pelanggan Starbucks jarang atau tidak pernah mengunjungi Starbucks.')
        st.write('========================================================================================')

        st.write('visualisasi 8')
        visualisasi8(df)
        st.write('Berdasarkan pie chart tersebut, dapat disimpulkan bahwa sebagian besar pelanggan Starbucks menghabiskan kurang dari RM40 per kunjungan. Hal ini menunjukkan bahwa Starbucks mungkin bukan tempat yang ideal bagi orang-orang yang ingin menghabiskan banyak uang untuk ke Starbucks.')
        st.write('========================================================================================')

        st.write('visualisasi 13')
        visualisasi13(df)
        st.write('Berdasarkan interpretasi grafik tersebut, dapat disimpulkan bahwa persentase pengguna yang memiliki kartu keanggotaan lebih tinggi daripada persentase pengguna yang tidak memiliki kartu keanggotaan.')
        st.write('========================================================================================')

        st.header('DISTRIBUTION')
        st.write('Visualisasi 1')
        visualisasi1(df) 
        st.write('Visualisasi ini menunjukkan bahwa Starbucks didominasi oleh pelanggan muda, dengan mayoritas berusia antara 20-39 tahun.')
        st.write('========================================================================================')

        st.header('RELATIONSHIP')
        st.write('visualisasi 4')
        visualisasi4(df)
        st.write('Grafik ini menunjukkan bahwa loyalitas pelanggan perempuan lebih tinggi daripada loyalitas pelanggan laki-laki. Pelanggan loyal perempuan dengan total pengeluaran lebih dari RM40 adalah yang paling banyak. Perusahaan dapat memanfaatkan informasi ini untuk mengembangkan strategi pemasaran yang lebih efektif untuk menargetkan pelanggan perempuan loyal dengan total pengeluaran yang tinggi.')
        st.write('========================================================================================')
        
        relationship(df2)
        st.write('========================================================================================')

        st.subheader('Kesimpulan')
        st.write('Setelah menganalisis berbagai visualisasi, beberapa kesimpulan dan insight yang dapat diambil adalah sebagai berikut:')
        st.write('1. Loyalitas Pelanggan: Loyalitas pelanggan Starbucks berkaitan dengan usia dan total pengeluaran. Pelanggan yang lebih muda, khususnya dalam rentang usia 30-39 tahun, dan memiliki total pengeluaran lebih dari RM40, cenderung menjadi pelanggan yang lebih loyal.')
        st.write('2. Perbedaan Loyalitas Gender: Perempuan cenderung lebih loyal daripada laki-laki dalam menjadi pelanggan Starbucks, terutama jika mereka memiliki total pengeluaran lebih dari RM40. Ini bisa menjadi wawasan berharga bagi perusahaan dalam mengarahkan strategi pemasaran mereka.')
        st.write('3. Membership: Pengguna dengan keanggotaan Starbucks cenderung memiliki tingkat loyalitas yang lebih tinggi dibandingkan dengan pengguna yang tidak memiliki keanggotaan. Oleh karena itu, promosi dan manfaat yang ditawarkan melalui kartu keanggotaan dapat membantu meningkatkan loyalitas pelanggan.')
        st.write('4. Frekuensi Kunjungan: Sebagian besar pelanggan Starbucks jarang atau bahkan tidak pernah mengunjungi Starbucks. Ini menunjukkan bahwa masih ada potensi untuk meningkatkan frekuensi kunjungan dengan menawarkan promosi atau insentif yang menarik.')
        st.write('5. Profil Pelanggan: Starbucks didominasi oleh karyawan dan pelajar, menunjukkan bahwa Starbucks merupakan tempat populer bagi orang-orang yang ingin bekerja atau belajar dengan suasana yang nyaman dan santai.')
        st.write('6. Profil Demografis: Pelanggan Starbucks didominasi oleh kelompok usia muda, terutama dalam rentang usia 20-39 tahun.')
        st.write('7. Total Pengeluaran: Sebagian besar pelanggan Starbucks menghabiskan kurang dari RM40 per kunjungan, menunjukkan bahwa Starbucks mungkin bukan tempat yang ideal bagi orang-orang yang ingin menghabiskan banyak uang.')
        st.write('8. Strategi Pemasaran: Informasi ini dapat digunakan oleh perusahaan untuk mengembangkan strategi pemasaran yang lebih efektif untuk menargetkan pelanggan yang paling mungkin menjadi loyal, seperti pelanggan perempuan dengan total pengeluaran yang tinggi.')
        st.write('Dengan memahami profil dan perilaku pelanggan dengan lebih baik, Starbucks dapat mengoptimalkan pengalaman pelanggan dan meningkatkan loyalitas pelanggan secara keseluruhan.')

    if __name__ == "__main__":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        main()

if selected == 'Columns':
    st.header('Interpretasi kolom yang terdapat dalam dataset "Starbucks Customer Survey"')
    df = pd.read_csv('Starbucks customer survey.csv')

    st.subheader('Columns in the Dataset')
    st.write(df.columns)

    st.subheader('1. Missing Values')
    st.write(df.isna().sum())
    st.write('Data Starbucks customer survey tersebut tidak memiliki missing values (nilai kosong) di dalamnya.')

    st.subheader('2. Duplicated Values')
    df[df.duplicated()]
    st.write('Data Starbucks customer survey tersebut tidak memiliki duplicated values (nilai duplikat) di dalamnya.')

    st.subheader('3. Outliers Values')
    results = []

    cols = df.select_dtypes(include=['float64', 'int64'])

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        percent_outliers = (len(outliers)/len(df))*100
        results.append({'Kolom': col, 'Persentase Outliers': percent_outliers})

    # Dataframe dari list hasil
    results_df = pd.DataFrame(results)
    results_df.set_index('Kolom', inplace=True)
    results_df = results_df.rename_axis(None, axis=0).rename_axis('Kolom', axis=1)

    st.dataframe(results_df)
    st.write('Data Starbucks customer survey tersebut memiliki outliers. Yakni pada kolom age, income, visitno, timespend, productrate, ambiancerate, wifirate, chooserate, promomethodeothers, dan loyal.')

    st.subheader('4. Penanganan Outliers Values')
    #Kolom Income
    Q1 = df['income'].quantile(0.25)
    Q3 = df['income'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['income'] >= lower_bound) & (df['income'] <= upper_bound)]

    #Kolom TimeSpend
    Q1 = df['timeSpend'].quantile(0.25)
    Q3 = df['timeSpend'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['timeSpend'] >= lower_bound) & (df['timeSpend'] <= upper_bound)]

    #Kolom WifiRate
    Q1 = df['wifiRate'].quantile(0.25)
    Q3 = df['wifiRate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['wifiRate'] >= lower_bound) & (df['wifiRate'] <= upper_bound)]

    #Kolom ChooseRate
    Q1 = df['chooseRate'].quantile(0.25)
    Q3 = df['chooseRate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['chooseRate'] >= lower_bound) & (df['chooseRate'] <= upper_bound)]

    #Kolom ProductRate
    Q1 = df['productRate'].quantile(0.25)
    Q3 = df['productRate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['productRate'] >= lower_bound) & (df['productRate'] <= upper_bound)]

    #Kolom PromoMethodOthers
    Q1 = df['promoMethodOthers'].quantile(0.25)
    Q3 = df['promoMethodOthers'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['promoMethodOthers'] >= lower_bound) & (df['promoMethodOthers'] <= upper_bound)]

    #Kolom AmbianceRate
    Q1 = df['ambianceRate'].quantile(0.25)
    Q3 = df['ambianceRate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['ambianceRate'] >= lower_bound) & (df['ambianceRate'] <= upper_bound)]

    results = []

    cols = df.select_dtypes(include=['float64', 'int64'])

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        percent_outliers = (len(outliers)/len(df))*100
        results.append({'Kolom': col, 'Persentase Outliers': percent_outliers})

    # Dataframe dari list hasil
    results_df = pd.DataFrame(results)
    results_df.set_index('Kolom', inplace=True)
    results_df = results_df.rename_axis(None, axis=0).rename_axis('Kolom', axis=1)

    st.dataframe(results_df)
    st.write('Namun setelah dilakukan penghapusan, maka kolom yang memiliki outliers sebanyak <10% telah di hapus outliersnya, sehingga datanya sudah tergolong bebas dari ooutliers.')

    st.subheader('5. Menghapus fitur-fitur yang tidak perlu')
    df = df.drop(['Id','itemPurchaseCoffee','itempurchaseCold', 'itemPurchasePastries', 'itemPurchaseJuices', 'itemPurchaseSandwiches', 'itemPurchaseOthers', 'chooseRate', 'promoMethodApp', 'promoMethodSoc', 'promoMethodEmail', 'promoMethodDeal', 'promoMethodFriend', 'promoMethodDisplay', 'promoMethodBillboard', 'promoMethodOthers'], axis=1)
    st.dataframe(df)
    st.write('Data diatas adalah data setelah tahap mengurangi fitur-fitur yang kurang relevan.')


# if selected == 'Classification':

#     # Load the dataset
#     # data = pd.read_csv('Starbucks customer survey.csv')

#     file_path = 'knn.pkl'

#     with open(file_path , 'rb') as f:
#         clf = joblib.load(f)

#     # Sidebar - Input features
    
#     age = st.selectbox('Age', ['<20', '20-29', '30-39', '40->40'])
#     income = st.selectbox('Income (RM)',  ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])
#     gender = st.selectbox('Gender', ['Male', 'Female']) 
#     status = st.selectbox('Occupation Status', ['Employed', 'Self-Employed', 'Student', 'Housewife'])
#     visitNo = st.selectbox('Number of Visits', ['Never', 'Daily', 'Weekly', 'Monthly'])
#     method = st.selectbox('Preferred Payment Method', ['Dine In', 'Drive Thru', 'Take Away', 'Never', 'Others'])
#     timeSpend = st.selectbox('Time Spent in Minutes', ['Below 30 mins', '30 mins to 1h', '1h to 2h', '2h to 3 h', ' More than 3h'])
#     location = st.selectbox('Preferred Location', ['Within 1km', '1km to 3km', 'More than 3km'])
#     membershipCard = st.selectbox('Membership Card', ['Yes', 'No'])
#     spendPurchase = st.selectbox('Average Spend per Purchase (RM)', ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])

#     prediction_state = st.markdown('PREDICT')

#     loyal = pd.DataFrame({
#         'Age': [age],
#         'Income (RM)': [income],
#         'Gender': [gender],
#         'Occupation Status': [status],
#         'Number of Visits': [visitNo],
#         'Preferred Payment Method': [method],
#         'Time Spent in Minutes': [timeSpend],
#         'Preferred Location': [location],
#         'Membership Card': [membershipCard],
#         'Average Spend per Purchase (RM)': [spendPurchase]
#     })

#     x_pred = clf.predict(loyal)

#     if x_pred[0] == 0:
#         msg = 'Pelanggan di prediksi Loyal'
#     else:
#         msg = 'Pelanggan di prediksi Tidak Loyal'

#     prediction_state.markdown(msg)


if selected == 'Classification':

    # Load the dataset
    data = pd.read_csv('Starbucks customer survey.csv')

    # Sidebar - Input features
    def user_input_features():
        age = st.selectbox('Age', ['<20', '20-29', '30-39', '40->40'])
        income = st.selectbox('Income (RM)',  ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])
        gender = st.selectbox('Gender', ['Male', 'Female']) 
        status = st.selectbox('Occupation Status', ['Employed', 'Self-Employed', 'Student', 'Housewife'])
        visitNo = st.selectbox('Number of Visits', ['Never', 'Daily', 'Weekly', 'Monthly'])
        method = st.selectbox('Preferred Payment Method', ['Dine In', 'Drive Thru', 'Take Away', 'Never', 'Others'])
        timeSpend = st.selectbox('Time Spent in Minutes', ['Below 30 mins', '30 mins to 1h', '1h to 2h', '2h to 3 h', ' More than 3h'])
        location = st.selectbox('Preferred Location', ['Within 1km', '1km to 3km', 'More than 3km'])
        membershipCard = st.selectbox('Membership Card', ['Yes', 'No'])
        spendPurchase = st.selectbox('Average Spend per Purchase (RM)', ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])

        data = {'Age': age,
                'Income (RM)': income,
                'Gender': gender,
                'Occupation Status': status,
                'Number of Visits': visitNo,
                'Preferred Payment Method': method,
                'Time Spent in Minutes': timeSpend,
                'Preferred Location': location,
                'Membership Card': membershipCard,
                'Average Spend per Purchase (RM)': spendPurchase}
        
        features = pd.DataFrame(data, index=[0])
        return features

    # Predict function
    def predict_classification(input_features, model):
        # Preprocess input features
        input_features = preprocess_data(input_features)
        
        # Make prediction
        prediction = model.predict(input_features)
        return prediction

    # Main function
    def main():
        st.title('Data Classification')
        st.subheader('Performing Data Classification')
        st.sidebar.header('User Input Features')
        
        user_input = user_input_features()
        
        if st.button('Predict'):
            # Train the model
            df_train = pd.read_csv('x_train.csv')
            model = df_train
            
            # Make prediction
            prediction = predict_classification(user_input, model)
            st.write("Prediction:", prediction)

    # Run the main function
    if __name__ == "__main__":
        main()
