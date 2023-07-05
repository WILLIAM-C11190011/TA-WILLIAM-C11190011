# TA-WILLIAM-C11190011
Kumpulan Kode Program Tugas Akhir William C11190011

File Augment.py digunakan untuk melakukan proses data augmentation pada dataset. Jumlah sampel kemasan tisu yang dimiliki jumlahnya sedikit, oleh sebab itu dilakukan proses data augmentation.
File process.py digunakan untuk memisahkan dataset untuk training dengan dataset untuk validation. Default pengaturan adalah 90% dari jumlah dataset untuk training dan 10% untuk validasi.
File sorting_ukuran.py digunakan untuk melakukan pendeteksian jarak kamera ke objek dengan memanfaatkan depth. File ini digunakan untuk mendeteksi ukuran kemasan tisu yang berbeda.
File testdeteksi.py digunakan untuk menjalankan pendeteksian dengan algoritma YOLOv4-Tiny hasil training. Pada program juga dilakukan pengecekan jarak kamera ke objek menggunakan depth untuk mendeteksi kejadian dimana terdapat kemasan yang tertumpuk
