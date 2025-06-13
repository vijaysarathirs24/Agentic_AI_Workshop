import streamlit as st
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ Must be FIRST Streamlit command
st.set_page_config(page_title="📘 Study Assistant", layout="centered")

# 🌙 Inject dark theme CSS with slide-down gradient effect
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom, #ffffff, #000000);
            color: #ffffff;
        }
        .stTextInput>div>div>input {
            color: #000000 !important;
        }
        .stTextArea textarea {
            color: #000000 !important;
        }
        .stButton>button {
            background-color: #222222 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Streamlit UI Header
st.title("📚 PDF Study Assistant")
st.markdown("Upload your course PDF to extract content, summarize it, and generate MCQs using Gemini AI.")

# 🔑 Gemini API Key Input Field
user_api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")

# 📄 Upload PDF
uploaded_file = st.file_uploader("📤 Upload PDF File", type=["pdf"])

# ✅ Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ✅ Generate summary from raw text
def generate_summary(text, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        max_tokens=500,
        google_api_key=api_key
    )

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize the following course material into clear bullet points for study preparation:\n\n{text}"""
    )

    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    return summary_chain.run(text=text)

# ✅ Generate MCQs from the summary
def generate_mcqs(summary, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        max_tokens=500,
        google_api_key=api_key
    )

    mcq_prompt = PromptTemplate(
        input_variables=["summary"],
        template="""Based on the following summary, generate 3 multiple-choice questions with 4 options each. Provide the correct answer too.\n\n{summary}\n\nFormat:\n- Question: [Your question here]\n  a) Option 1\n  b) Option 2\n  c) Option 3\n  d) Option 4\n  Correct Answer: [Correct option letter]"""
    )

    mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt)
    return mcq_chain.run(summary=summary)

# ✅ Main Flow
if uploaded_file is not None:
    if not user_api_key:
        st.warning("⚠️ Please enter your Gemini API key to proceed.")
    else:
        with st.spinner("📖 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        if text:
            st.success("✅ Text extracted successfully!")
            st.subheader("📄 Text Preview")
            st.text(text[:1000])  # Preview of extracted text

            if st.button("⚡ Generate Summary + MCQs"):
                with st.spinner("📚 Processing with Gemini..."):
                    try:
                        trimmed_text = text[:3000]
                        summary = generate_summary(trimmed_text, user_api_key)
                        questions = generate_mcqs(summary, user_api_key)

                        st.subheader("📝 Summary")
                        st.markdown(summary)

                        st.subheader("❓ Multiple Choice Questions")
                        st.text_area("Generated MCQs", questions, height=300)
                    except Exception as e:
                        st.error(f"❌ Error during generation: {e}")
        else:
            st.error("❌ Failed to extract readable content from the PDF.")
