# Eksperimen_SML_Nama-siswa

Folder ini mengikuti struktur Kriteria 1:

```text
Eksperimen_SML_Nama-siswa
├── .github/workflows/preprocessing.yml
├── namadataset_raw
└── preprocessing
    ├── Eksperimen_Nama-siswa.ipynb
    ├── automate_Nama-siswa.py
    └── namadataset_preprocessing
```

Jalankan preprocessing otomatis dari folder ini:

```bash
python preprocessing/automate_Nama-siswa.py --input namadataset_raw/hotel_bookings.csv --output preprocessing/namadataset_preprocessing
```

Notebook `preprocessing/Eksperimen_Nama-siswa.ipynb` berisi data loading, EDA, dan preprocessing manual sebagai sumber konversi ke `automate_Nama-siswa.py`.
