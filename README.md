# 📄 Smart Document Q&A Assistant (RAG Pipeline)

Aplikasi asisten berbasis AI yang memungkinkan Anda mengunggah dokumen PDF dan mengajukan pertanyaan spesifik secara interaktif. Dibangun menggunakan arsitektur **Retrieval-Augmented Generation (RAG)** modern dengan **LangChain**, model embedding lokal **HuggingFace**, database vektor **ChromaDB**, model bahasa **Google Gemini**, dan antarmuka web interaktif **Streamlit**.

---

## ✨ Fitur Utama

- **📄 Ekstraksi & Chunking PDF Cerdas**: Dokumen PDF diurai menggunakan `PyPDFLoader` dan dipotong menjadi segmen-segmen semantik menggunakan `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200).
- **🧠 Embedding Lokal Cepat & Hemat**: Menggunakan model `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace yang berjalan secara lokal (bebas kuota token embedding dan mendukung cache offline).
- **⚡ Chroma Vector Database**: Menyimpan dan mencari potongan dokumen yang paling relevan (*Top-K similarity retrieval*) secara cepat.
- **🤖 Google Gemini dengan Auto-Fallback**: Terintegrasi dengan model generasi teks Google Gemini (`gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`) dengan mekanisme otomatis beralih model jika server sedang padat (mengantisipasi *error 503*).
- **🛡️ Anti-Halusinasi & Sitasi Akurat**: Menggunakan *system prompt* terarah agar jawaban hanya berasal dari isi dokumen asli. Dilengkapi panel referensi yang menampilkan potongan teks asli beserta nomor halaman dokumen.
- **💻 Antarmuka Interaktif Streamlit**: Tampilan UI yang bersih, responsif, dan mudah digunakan langsung dari peramban web (*browser*).

---

## 📁 Struktur Direktori

```text
smart-doc-rag/
├── sample_docs/            # Folder untuk menyimpan contoh dokumen PDF uji coba
│   ├── .gitkeep
│   └── README.md
├── .env.example            # Template konfigurasi environment variable
├── .gitignore              # Daftar berkas & folder yang diabaikan oleh Git
├── app.py                  # Skrip utama aplikasi Streamlit & alur RAG
├── README.md               # Dokumentasi lengkap proyek
└── requirements.txt        # Daftar pustaka/dependensi Python
```

---

## 🔄 Alur Kerja Pipeline (RAG)

1. **Unggah Dokumen**: Pengguna mengunggah berkas PDF melalui panel sidebar.
2. **Text Splitting**: Dokumen diekstrak dan dipecah menjadi potongan-potongan teks (*chunks*) berukuran 1000 karakter dengan tumpang tindih (*overlap*) 200 karakter.
3. **Vektorisasi**: Setiap potongan teks dikonversi menjadi representasi vektor numerik menggunakan `all-MiniLM-L6-v2`.
4. **Indeksasi Chroma**: Vektor disimpan ke dalam *Vector Store* Chroma untuk pencarian instan.
5. **Pertanyaan Pengguna**: Saat pengguna mengajukan pertanyaan, sistem mencari 3 potongan teks paling relevan (*Top-3 retrieval*).
6. **Sintesis Konteks & LLM**: Konteks yang relevan digabungkan ke dalam *prompt anti-halusinasi* dan diteruskan ke Google Gemini untuk menghasilkan jawaban faktual dengan referensi halaman.

---

## 🚀 Panduan Memulai

### 1. Prasyarat Sistem
- **Python 3.10** atau versi lebih baru
- Koneksi internet (untuk mengunduh paket dan mengakses API Google Gemini)
- API Key Google Gemini (gratis di [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Kloning Repository (atau Navigasi ke Folder Proyek)
```bash
git clone https://github.com/<username-anda>/smart-doc-rag.git
cd smart-doc-rag
```

### 3. Buat dan Aktifkan Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 5. Konfigurasi Environment Variables
Salin berkas `.env.example` menjadi `.env`:
- **Windows (PowerShell):**
  ```powershell
  Copy-Item .env.example .env
  ```
- **Linux/macOS:**
  ```bash
  cp .env.example .env
  ```

Buka berkas `.env` lalu masukkan API Key Google Gemini Anda:
```env
GOOGLE_API_KEY="AIzaSyYourSecretKeyHere..."
```

### 6. Menjalankan Aplikasi
Jalankan perintah berikut di terminal:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Pustaka / Layanan | Kegunaan |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Dashboard & antarmuka tanya jawab interaktif |
| **Framework RAG** | [LangChain](https://www.langchain.com/) | Orkestrasi alur dokumen, prompt, dan model |
| **Document Loader** | `pypdf` via LangChain | Ekstraksi teks dari berkas PDF |
| **Embedding Model** | [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Vektorisasi teks lokal (`all-MiniLM-L6-v2`) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Penyimpanan vektor & *similarity search* |
| **LLM Provider** | [Google Generative AI](https://ai.google.dev/) | Inferensi jawaban dengan Gemini Flash |
| **Security & Env** | `python-dotenv`, `certifi` | Pengelolaan API Key & sertifikat SSL Windows |

---

## 🔒 Catatan Keamanan
- Berkas `.env` telah dimasukkan ke dalam `.gitignore` sehingga API key Anda **tidak akan** terunggah secara publik ke GitHub.
- Jangan pernah mempublikasikan file `.env` atau menyertakan API key langsung di dalam kode program.
