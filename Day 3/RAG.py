import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Optional Gemini integration
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- UI Setup ---
st.set_page_config(page_title="RAG QA System with Gemini AI", layout="wide")
st.title("\U0001F4D6 RAG-based Question Answering System")
st.caption("Upload PDFs and ask questions. Optionally use Gemini AI for faster, smarter answers.")

# --- Gemini API Setup ---
GEMINI_API_KEY = st.sidebar.text_input("place your gemini api for fast search: ", type="password")
if GEMINI_API_KEY and genai:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
else:
    gemini_model = None

# --- Helper: Call Gemini ---
def call_gemini_ai(prompt, max_tokens=150):
    if not gemini_model:
        return None, "Gemini AI not configured. Provide API key."
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7
            )
        )
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gemini API error: {str(e)}"

# --- PDF Loading ---
def load_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

# --- Chunking ---
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Embedding ---
def vectorize_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings, chunks

# --- Retrieval ---
def retrieve_relevant_chunks(query, embeddings, chunks, model, top_k=3):
    expanded_query = query
    if gemini_model:
        prompt = f"Rephrase this query to improve retrieval: {query}"
        expanded, error = call_gemini_ai(prompt, max_tokens=50)
        if not error: expanded_query = expanded
        else: st.warning(error)

    query_embedding = model.encode([expanded_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_indices]

# --- Answer Generation ---
def generate_answer(query, chunks, use_gemini=False):
    context = "\n".join(chunk for chunk, _ in chunks)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    if use_gemini and gemini_model:
        answer, error = call_gemini_ai(prompt)
        if error: st.error(error)
        return answer or "", context
    else:
        generator = pipeline("text-generation", model="distilgpt2")
        response = generator(prompt, max_length=150, num_return_sequences=1)
        return response[0]['generated_text'].strip(), context

# --- Citations ---
def source_attribution(chunks):
    return [f"Source {i+1}: {chunk[:100]}... (Score: {score:.4f})" for i, (chunk, score) in enumerate(chunks)]

# --- Core Execution ---
def run_rag(query, files, use_gemini=False):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_chunks, all_embeddings = [], []

    for file in files:
        text = load_pdf(file)
        if not text:
            st.warning(f"No text found in {file.name}")
            continue
        chunks = chunk_text(text)
        embeddings, _ = vectorize_chunks(chunks)
        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    if not all_embeddings:
        return None, None

    all_embeddings = np.vstack(all_embeddings)
    relevant = retrieve_relevant_chunks(query, all_embeddings, all_chunks, model)
    answer, _ = generate_answer(query, relevant, use_gemini=use_gemini)
    return answer, source_attribution(relevant)

# --- Streamlit UI ---
query = st.text_input("\U0001F50D Enter your question:", "What is positional encoding?")
files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
use_gemini = st.sidebar.checkbox("Use Gemini AI (if configured)", value=False)

if st.button("\U0001F4AC Get Answer"):
    if not files or not query:
        st.error("Upload PDFs and enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer, citations = run_rag(query, files, use_gemini=use_gemini)
            if answer:
                st.subheader("\U0001F4DD Answer")
                st.write(answer)
                st.subheader("\U0001F4C4 Sources")
                for cite in citations:
                    st.markdown(f"- {cite}")
            else:
                st.error("Could not generate an answer. Check file content or try another question.")
