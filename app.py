import os
import certifi

# Paksa httpx2 menggunakan bundel sertifikat CA certifi (mengatasi error SSL Windows)
os.environ["SSL_CERT_FILE"] = certifi.where()

import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Muat environment variables dari file .env
load_dotenv()

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Smart Doc RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 Smart Document Q&A (RAG Pipeline)")
st.caption("Unggah dokumen PDF dan ajukan pertanyaan berbasis konteks yang akurat.")


# Inisialisasi Model Embedding (Baca cache lokal terlebih dahulu agar instan)
@st.cache_resource
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


embeddings = load_embeddings()


# Fungsi Memproses PDF menjadi Vector Store
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    return vectorstore, len(docs), len(splits)


# Sidebar untuk Upload Dokumen
with st.sidebar:
    st.header("1. Unggah Dokumen")
    uploaded_file = st.file_uploader("Pilih file PDF", type=["pdf"])

    if uploaded_file is not None:
        if (
            "vectorstore" not in st.session_state
            or st.session_state.get("file_name") != uploaded_file.name
        ):
            with st.spinner("Mengekstrak teks & membangun Vector Index..."):
                vs, total_pages, total_chunks = process_pdf(uploaded_file)
                st.session_state.vectorstore = vs
                st.session_state.file_name = uploaded_file.name
                st.success(
                    f"Selesai! {total_pages} halaman dipecah menjadi {total_chunks} chunks."
                )

# Area Utama Tanya Jawab
if "vectorstore" in st.session_state:
    st.subheader(f"💬 Tanya Jawab: {st.session_state.file_name}")
    user_question = st.text_input("Masukkan pertanyaan spesifik mengenai isi dokumen:")

    if user_question:
        with st.spinner("Mencari referensi & menyusun jawaban..."):
            try:
                # 1. Ambil 3 potongan dokumen paling relevan (Retrieval)
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                retrieved_docs = retriever.invoke(user_question)
                context_text = "\n\n".join(
                    doc.page_content for doc in retrieved_docs
                )

                # 2. Susun Prompt Anti-Halusinasi
                system_prompt = (
                    "Anda adalah asisten analisis dokumen yang presisi. "
                    "Gunakan potongan konteks berikut untuk menjawab pertanyaan pengguna. "
                    "Jika jawaban tidak ditemukan di dalam konteks, katakan secara jujur bahwa "
                    "informasi tersebut tidak tersedia di dokumen. Jangan mengarang jawaban.\n\n"
                    "Konteks:\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )
                messages = prompt.format_messages(
                    context=context_text, input=user_question
                )

                # 3. Panggil LLM dengan Auto-Fallback berantai jika salah satu server sibuk (503)
                candidate_models = [
                    ("gemini-3.8-flash", {"thinking_level": "low"}),
                    ("gemini-3.6-flash", {}),
                    ("gemini-3.5-flash-lite", {}),
                ]

                response = None
                used_model = None
                last_error = None

                for model_name, extra_kwargs in candidate_models:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=model_name,
                            temperature=0.2,
                            timeout=20,
                            max_retries=1,
                            **extra_kwargs,
                        )
                        response = llm.invoke(messages)
                        used_model = model_name
                        break
                    except Exception as model_err:
                        last_error = model_err
                        continue

                if response is None:
                    raise last_error

                answer_text = (
                    response.content
                    if isinstance(response.content, str)
                    else "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in response.content
                    )
                )

                # Tampilkan Jawaban Utama
                st.markdown("### Jawaban:")
                st.caption(f"Model aktif: `{used_model}`")
                st.write(answer_text)

                # Tampilkan Sumber Kutipan
                with st.expander("🔍 Lihat Referensi Sumber (Chunks & Halaman)"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Referensi {i} — Halaman {page_num}:**")
                        st.info(doc.page_content)

            except Exception as e:
                st.error(f"Gagal memproses jawaban: {e}")
else:
    st.info("👈 Silakan unggah file PDF pada panel di sebelah kiri untuk memulai.")