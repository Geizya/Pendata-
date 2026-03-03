# Pendata-

## Setup dan Build Jupyter Book
Proyek ini menggunakan **Jupyter Book v1.x** (klasik) untuk menyajikan materi.
Berikut langkah cepat demi mereplikasi lingkungan:  

1. Buat/aktivasi virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   # .\\.venv\\Scripts\\Activate.ps1  (PowerShell/Windows)
   ```
2. Install dependensi:
   ```bash
   pip install -r materi-pendat/requirements.txt
   ```
3. Buat kerangka buku jika belum ada:
   ```bash
   jupyter-book create materi-pendat
   ```
4. Salin/letakkan konten markdown ke dalam folder `materi-pendat`.
   Misalnya `pertemuan2.md` dan `pertemuan3.md`.
5. Bangun situs:
   ```bash
   jupyter-book build materi-pendat
   ```
   Hasil terletak di `materi-pendat/_build/html/index.html`.

### Deployment ke GitHub Pages
Setelah build:  
```bash
pip install ghp-import
ghp-import -n -p -f materi-pendat/_build/html
```
Kemudian tunggu sampai halaman tersedia di `https://<username>.github.io/<repo>`.

Semua file asli (`materi/Pertemuan2`, dataset, workflow Orange, dsb.) dipertahankan—tidak ada yang dihapus.