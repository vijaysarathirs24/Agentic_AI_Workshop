import streamlit as st
import os
import requests
import base64
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

# Set up API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyAN11nQIl3Vw0D6bl3qBrQKdpG3Wkg9oFk"
os.environ["TAVILY_API_KEY"] = "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"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_output_tokens=2030)

# Sample job description dataset for RAG (fallback if Tavily fails)
job_descriptions = [
    "Software Engineer: Requires Python, Django, SQL. Experience in web development and APIs.",
    "Data Scientist: Needs Python, TensorFlow, data analysis. Projects in ML/AI preferred.",
    "Frontend Developer: Expertise in JavaScript, React, CSS. Strong UI/UX skills.",
]

# Set up vector store for RAG
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
job_docs = text_splitter.create_documents(job_descriptions)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(job_docs, embeddings)

# Project Analysis Agent
project_analysis_prompt = PromptTemplate(
    input_variables=["project_input"],
    template="Analyze the following input (GitHub README, resume, or project writeup) and extract: 1) Core technologies used, 2) Individual contributions, 3) Real-world impact. Input: {project_input}\nOutput format:\nTechnologies: \nContributions: \nImpact: ",
)
project_analysis_chain = project_analysis_prompt | llm | StrOutputParser()

def project_analysis_agent(project_input):
    return project_analysis_chain.invoke({"project_input": project_input})

# Role Matching Agent (RAG-Enabled with Tavily)
role_matching_prompt = PromptTemplate(
    input_variables=["project_analysis", "retrieved_jobs"],
    template="Based on the project analysis:\n{project_analysis}\nand job descriptions:\n{retrieved_jobs}\nIdentify the best matching entry-level tech job roles for freshers in 2025 and explain why they align with the student's skills and project scope.\nOutput format:\nBest Roles: \nAlignment Reasons: ",
)
role_matching_chain = role_matching_prompt | llm | StrOutputParser()

def role_matching_agent(project_analysis):
    # Retrieve job descriptions (try Tavily first, fallback to FAISS)
    try:
        tavily_tool = TavilySearchResults(k=3)
        job_result = tavily_tool.invoke(f"Top entry-level tech job roles for freshers in 2025 matching skills in {project_analysis}")
        retrieved_jobs = "\n".join([result.get('content', '') for result in job_result])
    except Exception as e:
        st.warning(f"Tavily search failed: {e}. Using fallback job descriptions.")
        retrieved_docs = vector_store.similarity_search(project_analysis, k=2)
        retrieved_jobs = "\n".join([doc.page_content for doc in retrieved_docs])
    return role_matching_chain.invoke({"project_analysis": project_analysis, "retrieved_jobs": retrieved_jobs})

# Content Refinement Agent
content_refinement_prompt = PromptTemplate(
    input_variables=["project_analysis", "matched_roles", "feedback_data"],
    template="Using the project analysis:\n{project_analysis}\nmatched roles:\n{matched_roles}\nand recruiter feedback:\n{feedback_data}\nGenerate compelling, role-specific descriptions for a resume, LinkedIn profile, a recruiter pitch line, and a short LinkedIn post (under 280 characters). Also suggest 3-5 tech companies hiring freshers in 2025 for these roles, with reasons for fit.\nOutput format:\nResume Description: \nLinkedIn Description: \nPitch Line: \nLinkedIn Post: \nTarget Companies: ",
)
content_refinement_chain = content_refinement_prompt | llm | StrOutputParser()

def content_refinement_agent(project_analysis, matched_roles, feedback_data):
    return content_refinement_chain.invoke({"project_analysis": project_analysis, "matched_roles": matched_roles, "feedback_data": feedback_data})

# Alignment Feedback Agent
alignment_feedback_prompt = PromptTemplate(
    input_variables=["project_analysis", "matched_roles", "refined_content", "feedback_data"],
    template="Analyze the project analysis:\n{project_analysis}\nmatched roles:\n{matched_roles}\nrefined content:\n{refined_content}\nand recruiter feedback:\n{feedback_data}\nProvide feedback on articulation gaps, company fit, and recommendations to improve presentation based on 2025 recruiter expectations. Also provide 1-2 placement tips for articulating projects in interviews.\nOutput format:\nArticulation Gaps: \nCompany Fit: \nRecommendations: \nPlacement Tips: ",
)
alignment_feedback_chain = alignment_feedback_prompt | llm | StrOutputParser()

def alignment_feedback_agent(project_analysis, matched_roles, refined_content, feedback_data):
    return alignment_feedback_chain.invoke({"project_analysis": project_analysis, "matched_roles": matched_roles, "refined_content": refined_content, "feedback_data": feedback_data})

# Skill Enhancement Agent (New)
skill_enhancement_prompt = PromptTemplate(
    input_variables=["project_analysis", "matched_roles", "feedback_data"],
    template="Based on the project analysis:\n{project_analysis}\nmatched roles:\n{matched_roles}\nand recruiter feedback:\n{feedback_data}\nIdentify skill gaps and recommend specific courses or certifications to improve the user's chances for the matched roles. Include course names, platforms (e.g., Coursera, Udemy), and reasons for relevance.\nOutput format:\nSkill Gaps: \nRecommended Courses: ",
)
skill_enhancement_chain = skill_enhancement_prompt | llm | StrOutputParser()

def skill_enhancement_agent(project_analysis, matched_roles, feedback_data):
    return skill_enhancement_chain.invoke({"project_analysis": project_analysis, "matched_roles": matched_roles, "feedback_data": feedback_data})

# Streamlit UI
st.set_page_config(page_title="Placement Match Maximizer", layout="wide")
st.title("🎯 Placement Match Maximizer")
st.write("A tool to help students align their projects with job roles effectively.")

# Input section
st.header("Step 1: Enter Project Details")
input_type = st.selectbox("Choose Input Type", ["Resume (PDF/TXT)", "Project Writeup", "GitHub Link"])
user_input = ""

if input_type == "Resume (PDF/TXT)":
    uploaded = st.file_uploader("Upload Resume", type=["pdf", "txt"])
    if uploaded:
        if uploaded.name.endswith(".pdf"):
            try:
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
    user_input = st.text_area("Paste your project writeup here:", height=200)
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
            st.error(f"❌ GitHub fetch failed: {str(e)}")

# Run Analysis
if st.button("🔍 Analyze and Process"):
    if user_input.strip():
        with st.spinner("Processing..."):
            # Step 1: Project Analysis
            st.header("Project Analysis")
            project_analysis_result = project_analysis_agent(user_input)
            st.markdown(project_analysis_result)

            # Step 2: Fetch Recruiter Feedback
            try:
                tavily_tool = TavilySearchResults(k=3)
                feedback_result = tavily_tool.invoke(f"Recruiter expectations for freshers in tech roles like Software Engineer, Data Scientist in 2025")
                feedback_data = "\n".join([result.get('content', '') for result in feedback_result])
            except Exception as e:
                st.warning(f"Tavily feedback search failed: {e}. Using default feedback.")
                feedback_data = "Recruiters expect clear articulation of project impact, strong technical skills, and alignment with company needs."

            # Step 3: Role Matching
            st.header("Role Matching")
            role_matching_result = role_matching_agent(project_analysis_result)
            st.markdown(role_matching_result)

            # Step 4: Content Refinement
            st.header("Content Refinement")
            content_refinement_result = content_refinement_agent(project_analysis_result, role_matching_result, feedback_data)
            st.markdown(content_refinement_result)

            # Step 5: Alignment Feedback
            st.header("Alignment Feedback")
            alignment_feedback_result = alignment_feedback_agent(project_analysis_result, role_matching_result, content_refinement_result, feedback_data)
            st.markdown(alignment_feedback_result)

            # Step 6: Skill Enhancement
            st.header("Skill Enhancement")
            skill_enhancement_result = skill_enhancement_agent(project_analysis_result, role_matching_result, feedback_data)
            st.markdown(skill_enhancement_result)
    else:
        st.error("Please provide valid input to proceed.")