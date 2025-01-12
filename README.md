# Laporan Proyek Machine Learning - Junathan Richie

## Project Overview
Berdasarkan data dari [Luminate](https://luminatedata.com/), pada tahun 2023, pada setiap hari terdapat 120.000 track baru yang ditambahkan ke dalam aplikasi _streaming music_. Dengan jumlah pertambahan sebanyak itu, pendengar musik akan kesulitan untuk mencari musik yang sesuai dengan seleranya. Selain itu, menyuguhkan lagu yang sesuai dengan minat pengguna juga menjadi salah satu faktor terbesar dalam persaingan dari aplikasi-aplikasi _streaming music_ [(Narici, 2020)](https://repositorio.ucp.pt/bitstream/10400.14/39880/1/202531554.pdf). 
<br>
<br>
Berdasarkan latar belakang tersebut, proyek ini akan membahas mengenai _recommendation system_ berbasis _content-based filtering_ untuk memberikan top-N rekomendasi musik dari suatu musik berdasarkan kriteria dari musik tersebut. Dataset yang digunakan dalam proyek ini adalah dataset 30000 Spotify Songs yang menyimpan informasi karakteristik _track-track_ dari aplikasi Spotify seperti _genre_, _subgenre_, _danceability_, _energy_, dan lainnya. Proyek ini diharapkan dapat menjadi dasar untuk mengembangkan model-model AI yang berkaitan dengan _recommendation system_ musik. 
<br>

## Business Understanding

### Problem Statements
Rumusan masalah berdasarkan latar belakang di atas sebagai berikut. 
- Bagaimana meningkatkan pengalaman pengguna dengan memberikan rekomendasi musik yang relevan menggunakan pendekatan content-based filtering?
- Bagaimana algoritma model yang tepat untuk memberikan top-N rekomendasi musik?

### Goals
Tujuan dari proyek ini adalah sebagai berikut: 
- mendapatkan cara untuk meningkatkan pengalaman pengguna dengan memberikan rekomendasi musik yang relevan menggunakan pendekatan content-based filtering
- mendapatkan algoritma model yang tepat untuk memberikan top-N rekomendasi musik 

### Solution statements
Cara yang dapat dilakukan untuk meraih goals tersebut adalah sebagai berikut:
- melakukan eksplorasi data untuk memahami data-data yang ada
- melakukan _preprocessing data_ agar sesuai dengan model yang ingin dibuat
- menggunakan algoritma Euclidean Distance untuk membuat model _content-based filtering_
- menggunakan algoritma Cosine Similarity untuk membuat model _content-based filtering_

## Data Understanding
Dataset yang digunakan adalah dataset _30000 Spotify Songs_ yang dibuat oleh Joakim Arvidsson dan dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs). Dataset ini terdiri dari 30.000 data track lagu dari Spotify dengan 23 fitur (kolom). Akan tetapi, karena keterbatasan komputasi dan RAM, proyek ini hanya akan mengambil 15.000 data dengan popularitas tertinggi. 

### Variabel-variabel pada 30000 Spotify Songs sebagai berikut:
- track_id: id unik dari lagu
- track_name: nama lagu
- track_artist: artis
- track_popularity: popularitas lagu dari range 1-100
- track_album_id: id unik dari album lagu
- track_album_name: nama dari album lagu
- track_album_release_date: tanggal rilis album
- playlist_name: nama dari playlist
- playlist_id: id playlist
- playlist_genre: genre dari playlist
- playlist_subgenre: subgenre dari playlist
- danceability: seberapa cocok sebuah lagu untuk ditarikan berdasarkan tempo, ritme, ketukan, dan lainnya. Nilai 1.0 berarti paling dapat ditarikan sedangkan 0.0 berarti tidak dapat ditarikan
- energy: energi diukur dalam rentang 0.0 hingga 1.0. Lagu yang energik terasa cepat, keras, dan berisik. Lagu dengan nilai energy tinggi seperti lagu death metal dan lagu dengan nilai energy rendah seperti Bach prelude.
- key: estimasi kunci keseluruhan track yang dipetakan dalam integer seperti 0 = C, 1 = C♯/D♭, 2 = D, dan seterusnya.
- loudness: kekerasan suara dari rata-rata seluruh track. Dalam rentang -60dB hingga 0dB.
- mode: menunjukkan modality (mayor atau minor) dari lagu. Mayor dilambangkan dengan 1 dan minor dilambangkan dengan 0.
- speechiness: menunjukkan seberapa banyak kata-kata yang diucapkan dalam track pada rentang 0.0 hingga 1.0. Semakin dekat dengan nilai 1.0 berarti semakin banyak kata yang diucapkan sehingga track itu mungkin saja sebuah podcast, audiobook, dan sebagainya. Nilai di atas 0,66 menggambarkan track yang mungkin seluruhnya terdiri dari kata-kata yang diucapkan. Nilai antara 0,33 dan 0,66 menggambarkan track yang mungkin berisi musik dan ucapan, baik dalam beberapa bagian atau berlapis, termasuk kasus seperti musik rap. Nilai di bawah 0,33 kemungkinan besar mewakili musik dan track lain yang tidak ada ucapan.
- acousticness: ukuran kepercayaan apakah sebuah lagu akustik, nilai 1.0 berarti sangat diyakini bahwa lagu itu adalah lagu akustik.
- instrumentalness: ukuran kepercayaan apakah sebuah lagu bersifat non-vocal (hanya instrumental). Semakin  mendekati nilai 1.0 berarti semakin diyakini bahwa track tersebut adalah instrumental
- liveness: liveness mendeteksi apakah ada penonton pada track itu. Apabila ada penonton, maka track tersebut cenderung dibuat secara live. Nilai di atas 0.8 menunjukkan bahwa sangat diakini bahwa track tersebut dibuat secara live.
- valence: digunakan untuk mengukur tingkat musical positiveness dari sebuah track. Track dengan nilai valence tinggi menunjukkan track terdengar positive (happy, cheerful) sedangkan nilai valence rendah menunjukkan track memiliki terdengar negative.
- tempo: perkiraan tempo dari keseluruhan track dalam BPM (beats per minute)
- duration_ms: durasi lagu dalam milliseconds

### Exploratory Data Analysis - Deskripsi Variabel
- Tipe data dari variabel
  | #   | Column                    | Non-Null Count  | Dtype    |
  | --- | ------------------------- | --------------- | -------- |
  | 0   | track_id                  | 15000 non-null  | object   |
  | 1   | track_name                | 15000 non-null  | object   |
  | 2   | track_artist              | 15000 non-null  | object   |
  | 3   | track_popularity          | 15000 non-null  | int64    |
  | 4   | track_album_id            | 15000 non-null  | object   |
  | 5   | track_album_name          | 15000 non-null  | object   |
  | 6   | track_album_release_date  | 15000 non-null  | object   |
  | 7   | playlist_name             | 15000 non-null  | object   |
  | 8   | playlist_id               | 15000 non-null  | object   |
  | 9   | playlist_genre            | 15000 non-null  | object   |
  | 10  | playlist_subgenre         | 15000 non-null  | object   |
  | 11  | danceability              | 15000 non-null  | float64  |
  | 12  | energy                    | 15000 non-null  | float64  |
  | 13  | key                       | 15000 non-null  | int64    |
  | 14  | loudness                  | 15000 non-null  | float64  |
  | 15  | mode                      | 15000 non-null  | int64    |
  | 16  | speechiness               | 15000 non-null  | float64  |
  | 17  | acousticness              | 15000 non-null  | float64  |
  | 18  | instrumentalness          | 15000 non-null  | float64  |
  | 19  | liveness                  | 15000 non-null  | float64  |
  | 20  | valence                   | 15000 non-null  | float64  |
  | 21  | tempo                     | 15000 non-null  | float64  |
  | 22  | duration_ms               | 15000 non-null  | int64    |

  Berdasarkan info tersebut, tipe data dari data adalah object, int64, dan float64. Data object ini berarti data yang berupa kategorial dan harus dikodekan nantinya. 

- Deskripsi dari setiap data

  | Statistic | track_popularity | danceability | energy   | key   | loudness   | mode  | speechiness | acousticness | instrumentalness | liveness | valence | tempo     | duration_ms |
  |-----------|------------------|--------------|----------|-------|------------|-------|-------------|--------------|------------------|----------|---------|-----------|-------------|
  | count     | 15000.00000      | 15000.000000 | 15000.000000 | 15000.000000 | 15000.000000 | 15000.000000 | 15000.000000  | 15000.000000   | 15000.000000      | 15000.000000 | 15000.000000 | 15000.000000 | 15000.000000  |
  | mean      | 64.80440         | 0.661734     | 0.681752 | 5.380667 | -6.540968  | 0.573200 | 0.107860    | 0.191325     | 0.051966         | 0.183591 | 0.520899 | 120.824316 | 219531.961067 |
  | std       | 10.79468         | 0.146650     | 0.177892 | 3.621653 | 2.916434   | 0.494629 | 0.101604    | 0.223007     | 0.177958         | 0.145301 | 0.227862 | 27.848624  | 53038.478476  |
  | min       | 48.00000         | 0.077100     | 0.000175 | 0.000000 | -46.448000 | 0.000000 | 0.022800    | 0.000002     | 0.000000         | 0.015000 | 0.000010 | 37.114000  | 45000.000000  |
  | 25%       | 56.00000         | 0.567000     | 0.567000 | 2.000000 | -7.875250  | 0.000000 | 0.040900    | 0.023600     | 0.000000         | 0.093100 | 0.346000 | 98.012000  | 186107.000000 |
  | 50%       | 63.00000         | 0.679000     | 0.704000 | 6.000000 | -5.960000  | 1.000000 | 0.063400    | 0.101000     | 0.000004         | 0.125000 | 0.522000 | 120.032000 | 212449.500000 |
  | 75%       | 72.00000         | 0.769000     | 0.817000 | 9.000000 | -4.566750  | 1.000000 | 0.132000    | 0.284000     | 0.000800         | 0.233250 | 0.699000 | 136.043750 | 244874.000000 |
  | max       | 100.00000        | 0.979000     | 1.000000 | 11.000000 | 0.551000   | 1.000000 | 0.918000    | 0.986000     | 0.994000         | 0.996000 | 0.990000 | 214.047000 | 517810.000000 |

  Hasil deskripsi dari data tidak menunjukkan keanehan pada data. 

### Exploratory Data Analysis - Data Analysis

- Track, Artis dan Album <br>
  ![image](https://github.com/user-attachments/assets/bc8eb948-ff2d-46bc-bbe9-5ee9055e9da4) <br>

  - Banyak track pada data: 28356
  - Banyak artis pada data: 10693
  - Banyak album pada data: 22545

- Popularity <br>
  ![image](https://github.com/user-attachments/assets/c6f4f387-fe29-428a-a383-dd9c4700c01c) <br>
  Terlihat bahwa sebagian besar data tersebar di antara 49-71. Data dengan popularity 95 ke atas sangat sedikit.
- Track Album Release Date (Year Distribution) <br>
  ![image](https://github.com/user-attachments/assets/9d6e6fef-85ce-4195-a1ca-5524f4e66e45) <br>
  Data di atas menunjukkan sebagian besar track berada pada tahun 2015 ke atas dan terbanyak pada tahun 2019.
- Playlist Genre <br>
  ![image](https://github.com/user-attachments/assets/15706341-1924-4fe8-a807-cccc77465e1a) <br>
  Perbandingan Genre terlihat didominasi oleh lagu pop. Lagu edm menjadi lagu dengan distribusi paling rendah.
- Playlist Sub-Genre <br>
  ![image](https://github.com/user-attachments/assets/2cc99ce2-8609-46eb-bc12-3aa8e1814006) <br>
  Sub-genre didominasi track-track seperti hip hop, urban contemporary, dan jenis-jenis pop. Sub-genre new jack swing dan big room memiliki pembagian paling kecil. 
- Duration <br>
  ![image](https://github.com/user-attachments/assets/3a5b5854-5bce-4b79-ad72-79b5920f8da9)<br>
  Sebagian besar track berada pada durasi 3-4 menit.
- Key <br>
  ![image](https://github.com/user-attachments/assets/fffddad6-5663-4f52-8fd9-9feac537a011) <br>
  Nada dasar C, C#, dan G menjadi nada dasar yang paling banyak digunakan dalam track lagu. Nada D# menjadi nada dasar yang paling sedikit digunakan. Hal ini menunjukkan distribusi preferensi kunci dari komposer ketika membuat lagu.
- Mode <br>
  ![image](https://github.com/user-attachments/assets/b8c11523-736a-4984-bac3-8b1b9012240b) <br>
  Mode menunjukkan lebih banyak track memiliki nada dasar major.
- Music Characteristic
  - Danceability <br>
    ![image](https://github.com/user-attachments/assets/b26a6b8d-892a-4a3c-833f-3905b673dfc1) <br>
    Penyebaran danceability yang condong data terpusat di kanan terutama di sekitar nilai 0.6 - 0.8 atau bisa disebut _negatively skewed distribution_ menggambarkan bahwa sebagian besar track memiliki sifat danceability yang tinggi. 
  - Energy <br>
    ![image](https://github.com/user-attachments/assets/cf19dd9c-c2ce-4bee-9dbf-dacbf7471bc9) <br>
    Penyebaran energy juga berbentuk _negatively skewed distribution_ dengan data terpusat di kanan di sekitar nilai 0.6 - 0.9. Sebagian besar track memiliki nilai energy yang tinggi. 
  - Loudness <br>
    ![image](https://github.com/user-attachments/assets/aee186e9-31e8-41c1-b565-6b535817c8b9)<br>
    Penyebaran loudness ekstrim di daerah -7.5 - -2.5 berarti sebagian besar track memiliki tingkat suara yang keras. 
  - Speechiness <br>
    ![image](https://github.com/user-attachments/assets/08a1e4fc-2fc4-4891-9204-605db0ece505) <br>
    Penyebaran speechiness ekstrim di bagian kiri di daerah 0 hingga 0.1 yang berarti sebagian besar track dinyanyikan dan sangat sedikit kata-kata yang diucapkan dalam track tersebut. 
  - Intrumentalness <br>
    ![image](https://github.com/user-attachments/assets/a327ee6d-77de-4e00-9bcc-96054a661dae) <br>
    Penyebaran instumentalness sangat ekstrim dengan 12000 lebih data berada di antara nilai 0.0 yang berarti lebih dari 3/4 data sangat diyakini bahwa track tersebut memiliki vocal di dalamnya. 
  - Liveness <br>
    ![image](https://github.com/user-attachments/assets/ce299589-cc8e-48aa-ab2e-e348662ba44e) <br>
    Liveness memiliki penyebaran data condong di daerah kiri di antara nilai 0.1 yang berarti sebagian besar track pada data tidak dinyanyikan secara live. 
  - Valence <br>
    ![image](https://github.com/user-attachments/assets/952ab376-3123-4acd-ad51-a21196bc7aa6) <br>
    Valence memiliki penyebaran yang merata. 
  - Acousticness <br> 
    ![image](https://github.com/user-attachments/assets/2f8eb19c-7236-4b76-b4ba-9941e8c4a6db)
    Acousticness memiliki penyebaran data ekstrim di sebelah kiri yang berarti sebagian besar track bukan lagu akustik. 
  - Tempo <br>
    ![image](https://github.com/user-attachments/assets/385a5b4d-5799-4166-9043-00508d637063)
    Tempo lagu tersebar cukup merata pada nilai 70-200 dan memiliki 2 puncak. Puncak pertama pada rentang data 90-100 dan puncak kedua pada rentang data 120-130. 
### Pengecekan Missing Value
Hasil pengecekan missing value menunjukkan bahwa tidak ada data null pada dataset. 
| Column                     | Missing Values |
|----------------------------|----------------|
| track_id                   | 0              |
| track_name                 | 0              |
| track_artist               | 0              |
| track_popularity           | 0              |
| track_album_id             | 0              |
| track_album_name           | 0              |
| track_album_release_date   | 0              |
| playlist_name              | 0              |
| playlist_id                | 0              |
| playlist_genre             | 0              |
| playlist_subgenre          | 0              |
| danceability               | 0              |
| energy                     | 0              |
| key                        | 0              |
| loudness                   | 0              |
| mode                       | 0              |
| speechiness                | 0              |
| acousticness               | 0              |
| instrumentalness           | 0              |
| liveness                   | 0              |
| valence                    | 0              |
| tempo                      | 0              |
| duration_ms                | 0              |
| duration_min               | 0              |

## Data Preparation
Data Preparation adalah tahap untuk memproses data sebelum digunakan untuk training model. Data preparation yang dilakukan pada proyek ini adalah:
- pengurangan jumlah data
- pengecekan dan drop duplicate
- drop kolom tidak relevan
- penyederhanaan data
- standarisasi data numerik
- encoding data kategori

### Pengurangan Jumlah Data
Pengurangan jumlah data ini adalah langkah pertama yang dilakukan pada data preparation. Pengurangan jumlah data dilakukan dengan hanya mengambil 15.000 data teratas pada dataset berdasarkan ```track_popularity```. Hal ini perlu dilakukan karena keterbatasan RAM dan komputasi pada proyek ini. 
```py
raw_data = raw_data.sort_values(by="track_popularity", ascending=False)
raw_data = raw_data.head(15000).reset_index(drop=True)
```
Pada notebook, langkah ini dilakukan sebelum Data Understanding agar hasil Data Understanding menunjukkan dengan jelas kondisi data hanya dari 15.000 data yang diambil tersebut. 

### Pengecekan dan Drop Duplicate
Data duplicate harus dihilangkan agar hasil rekomendasi tidak memberikan track yang sama dua kali serta meningkatkan efisiensi komputasi dengan menghilangkan perhitungan untuk data yang sama. Penghapusan data duplicate pada projek ini dilakukan dua kali sebagai berikut. 
1. Penghapusan data duplicate berdasarkan track_id <br>
  Berdasarkan pengecekan data duplicate berdasarkan track_id dengan
    ```py
    duplicates_by_track_id = raw_data[raw_data.duplicated(subset=['track_id'], keep=False)]
    duplicates_sorted_track_id = duplicates_by_track_id.sort_values(by='track_id')
    duplicates_sorted_track_id
    ```
    menunjukkan bahwa terdapat 5796 baris kolom dengan track_id duplicate. track_id duplicate juga memberikan data yang sama sehingga dapat diambil salah satu saja. 

2. Penghapusan data duplicate berdasarkan track_name dan track_artists<br>
  Sebuah track umumnya tidak memiliki judul dan nama artis yang sama. Seorang artis tidak umum untuk memiliki 2 lagu dengan judul yang sama. Oleh karena itu, sebuah track dengan track_name dan track_artists sama berarti track tersebut kemungkinan besar adalah track yang sama. Berdasarkan pengecekan dengan track_name dan track_artists didapatkan terdapat 1058 data dengan track_name dan track_artists yang sama. Akan tetapi, track sama ini memiliki tingkat popularity yang berbeda. Hal ini kemungkinan karena seorang artis mendaftarkan musiknya lebih dari satu kali dan salah satunya lebih populer. Oleh karena itu, track yang diambil adalah track yang memiliki nilai popularity lebih tinggi. 

### Drop Kolom Tidak Relevan
Kolom yang didrop pada dataset ini adalah: 
- track_id: tidak memberikan informasi mengenai karakteristik track dan hanya menjadi identifier sehingga dapat didrop
- data_track_album_id: metadata yang tidak digunakan dan tidak memberi informasi mengenai karakteristik track
- playlist_id: metadata yang tidak digunakan dan tidak memberi informasi mengenai karakteristik track
- data_track_album_name: metadata yang tidak digunakan dan tidak memberi informasi mengenai karakteristik track
- playlist_name: metadata yang tidak digunakan dan tidak memberi informasi mengenai karakteristik track
- track_popularity: tidak berpengaruh terhadap karakteristik musik sedangkan sistem rekomendasi yang dibuat akan berdasarkan karakteristrik dari musik sehingga dapat didrop
- liveness: tidak berpengaruh terhadap karakteristik musik sedangkan sistem rekomendasi yang dibuat akan berdasarkan karakteristrik dari musik sehingga dapat didrop
- duration: tidak berpengaruh terhadap karakteristik musik sedangkan sistem rekomendasi yang dibuat akan berdasarkan karakteristrik dari musik sehingga dapat didrop
- release_date: tidak berpengaruh terhadap karakteristik musik sedangkan sistem rekomendasi yang dibuat akan berdasarkan karakteristrik dari musik sehingga dapat didrop

### Penyederhanaan Data 
Data yang berhubungan dengan kepercayaan seperti acousticness dan instrumentalness diubah agar menjadi 0 dan 1. Hal ini untuk menyederhanakan perhitungan dengan menggunakan cosine similarity dan euclidean distance. Nilai acousticness dan instrumentalness akan diubah dengan kode berikut. 
```py
cleaned_data.loc[cleaned_data["acousticness"] <= 0.5, "acousticness"] = 0
cleaned_data.loc[cleaned_data["acousticness"] > 0.5, "acousticness"] = 1
cleaned_data.loc[cleaned_data["instrumentalness"] <= 0.5, "instrumentalness"] = 0
cleaned_data.loc[cleaned_data["instrumentalness"] > 0.5, "instrumentalness"] = 1
```
Nilai 0.5 ke bawah akan diubah menjadi 0 sedangkan nilai di atas 0.5 diubah menjadi 1. 

### Standarisasi Data Numerik
Standarisasi diperlukan agar model menganggap setiap fitur memiliki bobot yang sama dan tidak bias dengan rentang nilai yang lebih besar. Standarisasi yang dilakukan pada proyek ini adalah dengan menggunakan MinMaxScaler. MinMaxScaler bekerja dengan mengubah rentang nilai dari setiap fitur agar berada di antara 0 hingga 1. Hal ini penting untuk algoritma berbasis jarak seperti Euclidean Distance untuk memberikan hasil yang lebih akurat dan adil. MinMaxScaler diterapkan pada data numerik yaitu danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, valence, tempo, mode. 

### Encoding Data Kategori
Encoding data kategori diperlukan karena algoritma Euclidean Distance dan Cosine Similarity yang akan digunakan hanya dapat mengolah data numerik dan tidak dapat mengolah data kategorial sehingga perlu adanya encoding. Pada proyek ini, encoding data kategori dilakukan dengan menggunakan One Hot Encoder. One Hot Encoder bekerja dengan membuat kolom baru dari setiap kategori yang ada. Setiap kolom akan merepresentasikan satu kategori unik, dan nilainya diisi dengan 1 jika data termasuk dalam kategori tersebut, atau 0 jika tidak. Encoding data kategori yang dilakukan pada proyek ini hanya kepada subgenre saja. Hal ini karena subgenre sudah merepresentasikan genre dan bersifat lebih spesifik daripada genre. 

## Modeling
Modeling yang dilakukan pada sistem rekomendasi ini adalah dengan menggunakan _euclidean distance_ dan _cosine similarity_. 
### _Euclidean Distance_
#### Cara Kerja
- _Euclidean Distance_ bekerja dengan mengukur jarak lurus antara dua titik di ruang multidimensi. 
- Rumus dari _Euclidean Distance_ sebagai berikut. <br>
  ```dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))```
  
#### Tahap Penerapan Model
- Euclidean distance dari library sklearn.metrics.pairwise diimport lalu diterapkan pada dataframe yang telah dipreprocess sebelumnya. 
  ```py
  from sklearn.metrics.pairwise import euclidean_distances
  features = numerical_cols + list(encoded_df.columns)
  euclidean_dist = euclidean_distances(data_encoded[features])
  ```
- Karena euclidean distance menghasilkan matrix yang menggambarkan jarak antar data, sedangkan yang dibutuhkan adalah kedekatan antar data, data euclidean distance tersebut diubah agar mendapatkan kedekatannya. 
  ```py
  euclidean_sim = 1 / (1 + euclidean_dist)
  ```
- euclidean_sim tersebut berupa matrix berukuran (10949, 10949) yang sesuai dengan jumlah data pada ```data_encoded```. Nilai pada matrix tersebut berada pada rentang 0-1 yang menggambarkan kedekatan dari setiap pasangan data. 
  ```
  array([[1.        , 0.36157329, 0.40302967, ..., 0.37756056, 0.36461074,
          0.4108236 ],
        [0.36157329, 1.        , 0.36257567, ..., 0.34336881, 0.33599485,
          0.361447  ],
        [0.40302967, 0.36257567, 1.        , ..., 0.37704308, 0.3657713 ,
          0.39873561],
        ...,
        [0.37756056, 0.34336881, 0.37704308, ..., 1.        , 0.79824121,
          0.56443413],
        [0.36461074, 0.33599485, 0.3657713 , ..., 0.79824121, 1.        ,
          0.52159808],
        [0.4108236 , 0.361447  , 0.39873561, ..., 0.56443413, 0.52159808,
          1.        ]])
  ```

#### Kelebihan
- Sederhana dan mudah dipahami
- Cocok untuk data kontinuitas (memiliki hubungan numerik yang berkesinambungan)
- Representasi jarak yang bermakna, cocok dengan sistem rekomendasi yang akan dibuat karena jarak antar titik dihitung secara bermakna. Perbedaan antar nilai mencerminkan tingkat perbedaan secara langsung.
#### Kekurangan
- Peka terhadap skala sehingga membutuhkan standarisasi
- Kurang efektif untuk data berdimensi tinggi (_curse of dimensionality_), jarak antar titik menjadi hampir seragam dan kehilangan maknanya. 

### _Cosine Similarity_
#### Cara Kerja
- _Cosine Similarity_ bekerja dengan mengukur kesamaan antara dua vektor dengan menghitung cosinus sudut di antara kedua vektor tersebut. 
- Rumus dari _Cosine Similarity_ sebagai berikut. 
  ```
  K(X, Y) = <X, Y> / (||X||*||Y||)
  ```
#### Tahap Penerapan Model
- Cosine Similarity diterapkan pada dataframe data yang telah dipreproceessing sebelumnya dengan menggunakan cosine_similarity dari library sklearn.metrics.pairwise
  ```py
  from sklearn.metrics.pairwise import cosine_similarity
  cosine_sim = cosine_similarity(data_encoded[features])
  ```
- cosine_sim akan menghasilkan matrix berukuran (10949, 10949) dengan nilai -1 hingga 1 dengan 1 berarti menunjukkan kesamaan dan -1 menunjukkan ketidaksamaan antar data.
  ```
  array([[1.        , 0.62784331, 0.71924839, ..., 0.67930401, 0.66194426,
        0.74122375],
       [0.62784331, 1.        , 0.62587194, ..., 0.58735927, 0.58113649,
        0.62772224],
       [0.71924839, 0.62587194, 1.        , ..., 0.67360859, 0.66160429,
        0.70929622],
       ...,
       [0.67930401, 0.58735927, 0.67360859, ..., 1.        , 0.9942956 ,
        0.93112792],
       [0.66194426, 0.58113649, 0.66160429, ..., 0.9942956 , 1.        ,
        0.91046053],
       [0.74122375, 0.62772224, 0.70929622, ..., 0.93112792, 0.91046053,
        1.        ]])
  ```
#### Kelebihan
- Tidak dipengaruhi oleh perbedaan skala atau panjang vektor karena hanya fokus pada arah vektor.
- Cocok pada data sparsity (memiliki banyak 0)
- Efektif untuk menangkap kesamaan pola
#### Kekurangan
- Kurang cocok untuk data kontinuitas
- Hanya untuk menggambarkan arah 
- Tidak memperhatikan panjang (magnitudo) dari vektor

### Fungsi untuk Menghasilkan Top-N Recommendation
Fungsi berikut akan menerima parameter berupa track_name (judul lagu), track_artist (penyanyi), pairwise_metrics yang digunakan (seperti matrix hasil cosine similarity atau hasil euclidean distance), dataset yang digunakan, serta top_n sebagai jumlah hasil yang akan diambil. 
```py
def get_song_recommendations(track_name, track_artist, pairwise_metrics, dataset, top_n=10):
    # Mengecek apakah kombinasi track_name dan track_artist ada dalam dataset
    track_match = dataset[
        (dataset['track_name'] == track_name) & 
        (dataset['track_artist'].str.contains(track_artist, case=False))
    ]

    if track_match.empty:
        raise ValueError("Lagu tidak ditemukan atau artis tidak cocok.")

    # Mengambil indeks lagu berdasarkan kombinasi yang cocok
    track_idx = track_match.index[0]
    print(f"track_idx: {track_idx}")
    
    # Memastikan bahwa track_idx berada dalam rentang yang valid pada pairwise_metrics
    if track_idx >= pairwise_metrics.shape[0]:
        raise IndexError("Indeks lagu tidak valid dalam pairwise metrics.")

    # Mendapatkan nilai similarity untuk lagu yang dimasukkan
    sim_scores = list(enumerate(pairwise_metrics[track_idx]))

    # Mengurutkan similarity dari yang tertinggi
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Ambil sebanyak top_n tetapi exclude lagu yang dicari 
    sim_scores = sim_scores[1:top_n+1]

    # Mendapatkan indeks lagu yang direkomendasikan
    song_indices = [i[0] for i in sim_scores]
    recommended_songs = dataset.iloc[song_indices].copy()
    recommended_songs.loc[:, 'similarity_score'] = [score[1] for score in sim_scores]
    return recommended_songs
```
Percobaan fungsi ini akan menggunakan lagu Secukupnya dari Hindia yang memiliki karakteristik sebagai berikut. 

| track_name   | track_artist | playlist_genre | playlist_subgenre | danceability | energy   | key      | loudness | mode | speechiness | acousticness | instrumentalness | valence  | tempo    |
|--------------|--------------|----------------|-------------------|--------------|----------|----------|----------|------|-------------|--------------|------------------|----------|----------|
| Secukupnya   | Hindia       | latin          | latin pop         | 0.725025     | 0.795964 | 0.818182 | 0.879806 | 0.0  | 0.012958    | 0.0          | 0.0              | 0.942424 | 0.36676  |

Sebagai contoh jika track_name dalam fungsi adalah "Secukupnya" dengan track_artist "Hindia" jika pairwise_metrics adalah **euclidean_sim (algoritma Euclidean Distance)**, dan top_n 5 akan menghasilkan sebagai berikut. 

| track_name                     | track_artist       | playlist_genre | playlist_subgenre | danceability | energy   | key      | loudness | mode | speechiness | acousticness | instrumentalness | valence  | tempo    | similarity_score |
|--------------------------------|--------------------|----------------|-------------------|--------------|----------|----------|----------|------|-------------|--------------|------------------|----------|----------|------------------|
| Chantaje (feat. Maluma)        | Shakira            | latin          | latin pop         | 0.859186     | 0.772960 | 0.727273 | 0.926126 | 0.0  | 0.061215    | 0.0          | 0.0              | 0.916161 | 0.366919 | 0.848345         |
| Jam                            | Starboy            | latin          | latin pop         | 0.729460     | 0.760958 | 0.909091 | 0.898508 | 0.0  | 0.030049    | 0.0          | 0.0              | 0.797978 | 0.321687 | 0.846174         |
| DUELE EL CORAZON               | Enrique Iglesias   | latin          | latin pop         | 0.717264     | 0.903983 | 0.727273 | 0.916913 | 0.0  | 0.082440    | 0.0          | 0.0              | 0.854544 | 0.304550 | 0.837242         |
| Back In The City               | Alejandro Sanz     | latin          | latin pop         | 0.749418     | 0.753957 | 0.909091 | 0.854550 | 0.0  | 0.017538    | 0.0          | 0.0              | 0.861615 | 0.513629 | 0.834409         |
| No Te Veo - Digital Single     | Casa De Leones     | latin          | latin pop         | 0.853642     | 0.872978 | 0.727273 | 0.886040 | 0.0  | 0.064343    | 0.0          | 0.0              | 0.975757 | 0.445892 | 0.831970         |

Sedangkan apabila pairwise_metrics menggunakan **cosine_sim (algoritma Cosine Similarity)** hasilnya sebagai berikut. 
| track_name                     | track_artist       | playlist_genre | playlist_subgenre | danceability | energy   | key      | loudness | mode | speechiness | acousticness | instrumentalness | valence  | tempo    | similarity_score |
|--------------------------------|--------------------|----------------|-------------------|--------------|----------|----------|----------|------|-------------|--------------|------------------|----------|----------|------------------|
| Chantaje (feat. Maluma)        | Shakira            | latin          | latin pop         | 0.859186     | 0.772960 | 0.727273 | 0.926126 | 0.0  | 0.061215    | 0.0          | 0.0              | 0.916161 | 0.366919 | 0.996605         |
| Jam                            | Starboy            | latin          | latin pop         | 0.729460     | 0.760958 | 0.909091 | 0.898508 | 0.0  | 0.030049    | 0.0          | 0.0              | 0.797978 | 0.321687 | 0.996490         |
| No Te Veo - Digital Single     | Casa De Leones     | latin          | latin pop         | 0.853642     | 0.872978 | 0.727273 | 0.886040 | 0.0  | 0.064343    | 0.0          | 0.0              | 0.975757 | 0.445892 | 0.996351         |
| Lejos De Ti                    | Gian Marco         | latin          | latin pop         | 0.685109     | 0.754957 | 0.636364 | 0.842167 | 0.0  | 0.017650    | 0.0          | 0.0              | 0.777776 | 0.355519 | 0.996138         |
| DUELE EL CORAZON               | Enrique Iglesias   | latin          | latin pop         | 0.717264     | 0.903983 | 0.727273 | 0.916913 | 0.0  | 0.082440    | 0.0          | 0.0              | 0.854544 | 0.304550 | 0.995926         |

## Evaluation
Metrik evaluasi yang digunakan untuk proyek ini adalah _Precision_. Metrik ini cocok digunakan untuk algoritma _content-based filtering_. _Precision_ memiliki formula sebagai berikut. <br>

![image](https://github.com/user-attachments/assets/54152086-79de-4d4f-9637-c7e5705e2ea7)<br>

Berdasarkan tabel hasil dari **Euclidean Distance**, seluruh hasilnya memiliki genre, subgenre, mode, acousticness, dan instrumentalness yang sama serta selisih dari nilai-nilai numerik lainnya tidak jauh. Hal ini berarti hasil yang diberikan relevan sehingga memiliki _Precision_ 5/5 atau 100%. 
<br><br>
Hal ini juga sama dengan tabel hasil dari **Cosine Similarity**. Seluruh hasilnya memberikan genre, subgenre, mode, acousticness, dan istumentalness yang sama serta juga memiliki selisih nilai numerik lainnya yang tidak jauh. Dengan hal ini, dapat disimpulkan bahwa nilai _Precision_ juga 5/5 atau 100%. 
<br><br>
Catatan tambahan:
- Perbedaan besar pada similarity_score antara Euclidean Distance dengan Cosine Similarity diakibatkan perbedaan cara kerja kedua algoritma tersebut. Cosine Similarity melihat kesamaan dari vektor fitur sedangkan Euclidean Distance melihat jarak absolut antar ruang fitur. 

## Kesimpulan 
Kesimpulan dari proyek ini sebagai berikut.
- Sistem Rekomendasi berbasis _content-based filtering_ dapat digunakan untuk meningkatkan pengalaman pengguna berdasarkan mencari lagu yang mirip dengan lagu yang disukai oleh pendengar atau terakhir didengar oleh pendengar. Karakteristik lagu ini dapat menjadi acuan untuk mencari lagu lain yang serupa dari sisi karakteristik musiknya. 
- **Cosine Similarity** dan **Euclidean Distance** keduanya dapat digunakan untuk memberikan rekomendasi berdasarkan suatu track. Pemilihan antara menggunakan **Cosine Similarity** atau **Euclidean Distance** berdasarkan tujuannya. Jika tujuannya untuk mengukur kesamaan vektor fitur tanpa mempedulikan panjang vektor, **Cosine Similarity** lebih cocok. Sedangkan apabila untuk mengukur jarak nyata antara dua titik, **Euclidean Distance** lebih cocok. Untuk proyek ini sendiri, saya lebih cenderung memilih **Euclidean Distance** karena banyaknya data yang bersifat kontinu dan menghitung jarak antar titik memberikan hasil yang lebih presisi.

## Referensi 
[There are now 120,000 new tracks hitting music streaming services each day](https://www.musicbusinessworldwide.com/there-are-now-120000-new-tracks-hitting-music-streaming-services-each-day/) <br>
[Music Discovery on Online Streaming Platforms The Role of Consumers’ Subjective Expertise](https://repositorio.ucp.pt/bitstream/10400.14/39880/1/202531554.pdf)
