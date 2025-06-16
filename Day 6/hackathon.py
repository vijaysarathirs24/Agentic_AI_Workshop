import streamlit as st
import os
import requests
import base64
import pdfplumber  # Using pdfplumber instead of PyMuPDF for PDF text extraction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

# === Set API Keys ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyAN11nQIl3Vw0D6bl3qBrQKdpG3Wkg9oFk"
os.environ["TAVILY_API_KEY"] = "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"

# === UI ===
st.set_page_config(page_title="Placement Match Maximizer", layout="wide")
st.title("🎯 Placement Match Maximizer")

# === LLM Settings ===

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    max_output_tokens=2030
)
# === Prompt Template for LLM ===
rewrite_prompt = ChatPromptTemplate.from_template("""
You are a placement expert specializing in helping students and freshers. Based on the user input and the following job role descriptions, provide:

- Matching Job Roles: Identify entry-level tech job roles suitable for freshers that align with the user's demonstrated skills. Focus on roles that value personal projects and practical experience.
- Rewritten Resume/Project Bullet Points: Rewrite the user's resume or project bullet points to align with the matched job roles. When analyzing GitHub links or project writeups, focus on the user's personal projects, highlighting technical skills, problem-solving abilities, and impact.
- Company Targeting: Suggest 3-5 specific tech companies that are hiring freshers for the matched job roles in 2025. Provide a brief reason why each company is a good fit based on the user's skills and the company's focus.
- A Short LinkedIn Post (under 280 characters): Showcase the user's strengths, emphasizing their personal projects and skills relevant to the matched roles.
- Extra Suggestions: Recommend specific courses or certifications the user can take to improve their chances of landing the matched job roles. Include course names, platforms (e.g., Coursera, Udemy, LinkedIn Learning), and a brief reason why each course is relevant.
- Placement Tips: Provide 1-2 specific tips for articulating personal projects during campus placements or interviews (e.g., how to discuss the project’s impact or technical challenges).

Use the following recruiter feedback to guide your alignment recommendations: {feedback_data}

## USER INPUT:
{input}

## JOB ROLES:
{job_data}
""")

# === Agent for Tool Use (Tavily Search) ===
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful agent that finds current job roles and recruiter feedback for users."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
tavily_tool = TavilySearchResults(k=3)
agent = create_tool_calling_agent(llm=llm, tools=[tavily_tool], prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[tavily_tool], verbose=False)

# === Input Section ===
input_type = st.selectbox("Choose Input Type", ["Resume (PDF/TXT)", "Project Writeup", "GitHub Link"])
user_input = ""

if input_type == "Resume (PDF/TXT)":
    uploaded = st.file_uploader("Upload Resume", type=["pdf", "txt"])
    if uploaded:
        if uploaded.name.endswith(".pdf"):
            try:
                # Open the PDF using pdfplumber
                with pdfplumber.open(uploaded) as pdf:
                    user_input = ""
                    for page in pdf.pages:
                        user_input += page.extract_text() + "\n" if page.extract_text() else ""
                if not user_input.strip():
                    st.error("❌ No text could be extracted from the PDF.")
            except Exception as e:
                st.error(f"❌ Failed to extract text from PDF: {str(e)}")
        else:
            user_input = uploaded.read().decode("utf-8", errors="ignore")

elif input_type == "Project Writeup":
    user_input = st.text_area("Paste your project writeup here")

elif input_type == "GitHub Link":
    github_link = st.text_input("Enter your GitHub link")
    if github_link:
        try:
            repo = "/".join(github_link.strip("/").split("/")[-2:])
            api_url = f"https://api.github.com/repos/{repo}/readme"
            response = requests.get(api_url)
            if response.status_code == 200:
                content = base64.b64decode(response.json()['content']).decode()
                user_input = content[:3000]
                st.success("✅ README loaded from GitHub.")
                st.code(user_input[:1000])
            else:
                st.error("❌ Could not fetch GitHub README.")
        except Exception as e:
            st.error(str(e))

# === Run Analysis ===
if st.button("🔍 Match Jobs and Rewrite") and user_input.strip():
    with st.spinner("Fetching matching job roles..."):
        try:
            # Placeholder for skill extraction (in a real implementation, use NLP/keyword extraction)
            skills = ["Python", "data analysis", "machine learning"]  # Replace with actual skill extraction logic
            job_result = agent_executor.invoke({
                "input": f"List top 2025 tech job roles for freshers with skills in {', '.join(skills)}"
            })
            job_data = job_result['output']

            # Fetch recruiter feedback
            recruiter_feedback = agent_executor.invoke({
                "input": f"What do recruiters look for in freshers applying for roles like {', '.join(['Data Analyst', 'Software Developer'])} in 2025?"
            })
            feedback_data = recruiter_feedback['output']
        except Exception as e:
            st.error(f"Tavily search failed: {e}")
            job_data = ""
            feedback_data = ""

    with st.spinner("Generating customized suggestions..."):
        try:
            final_prompt = rewrite_prompt.format_messages(
                input=user_input,
                job_data=job_data,
                feedback_data=feedback_data
            )
            result = llm.invoke(final_prompt)
            st.subheader("📄 Personalized Resume Alignment")
            st.markdown(result.content)
        except Exception as e:
            st.error(f"Gemini generation failed: {e}")