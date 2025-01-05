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
- Bagaimana model yang tepat untuk memberikan top-N rekomendasi musik dengan _content-based filtering_?
- Bagaimana cara mengukur evaluasi model pada _content-based filtering_?

### Goals
Tujuan dari proyek ini adalah sebagai berikut: 
- mendapatkan model yang tepat untuk memberikan top-N rekomendasi musik dengan _content-based filtering_
- mendapatkan cara mengukur evaluasi model pada _content-based filtering_

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

- Track, Artis dan Album
  Gambar
  - Banyak track pada data: 28356
  - Banyak artis pada data: 10693
  - Banyak album pada data: 22545

- Popularity

- Track Album Release Date 

- Playlist Genre

- Playlist Sub-Genre

- Key

- Mode 
 
- Music Characteristic
  - Danceability
  - Energy
  - Loudness
  - Speechiness
  - Intrumentalness
  - Liveness
  - Valence
  - Acousticness
  - Tempo
## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

## Referensi 
[There are now 120,000 new tracks hitting music streaming services each day](https://www.musicbusinessworldwide.com/there-are-now-120000-new-tracks-hitting-music-streaming-services-each-day/) <br>
[Music Discovery on Online Streaming Platforms The Role of Consumers’ Subjective Expertise](https://repositorio.ucp.pt/bitstream/10400.14/39880/1/202531554.pdf)