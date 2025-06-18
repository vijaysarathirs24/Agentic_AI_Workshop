import os
import re
import json
import tempfile
from typing import List, Dict
import dotenv
from github import Github, GithubException
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import PyPDF2
import streamlit as st

# ----------------- 🔐 Load Environment -----------------
dotenv.load_dotenv()
GOOGLE_API_KEY = ("GOOGLE GEMINI TOKEN")
GITHUB_TOKEN = ("GIT_TOKEN")

if not GOOGLE_API_KEY or not GITHUB_TOKEN:
    st.error("Required API keys (Google/Github) not found in .env. Ensure .env file contains GOOGLE_API_KEY and GITHUB_TOKEN.")
    st.stop()

# ----------------- 🧠 Tool Functions -----------------

def parse_github_repo(repo_url: str) -> Dict:
    """Extracts details from a GitHub repository."""
    try:
        g = Github(GITHUB_TOKEN)
        repo_match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/|$)", repo_url)
        if not repo_match:
            return {"error": "Invalid GitHub URL format. Example: https://github.com/owner/repo"}
        repo_name = f"{repo_match.group(1)}/{repo_match.group(2)}"
        repo = g.get_repo(repo_name)
        readme = repo.get_readme().decoded_content.decode('utf-8', errors='ignore')[:2000] if repo.get_readme() else ""
        languages = repo.get_languages()
        contributors = [c.login for c in repo.get_contributors()]
        commits = repo.get_commits().totalCount
        return {
            "name": repo.name,
            "description": repo.description or "No description provided",
            "readme": readme,
            "languages": languages,
            "individual_contribution": f"Contributor among {len(contributors)} in {commits} commits",
            "impact": f"Stars: {repo.stargazers_count}, Forks: {repo.forks_count}, Issues: {repo.open_issues_count}"
        }
    except GithubException as e:
        return {"error": f"GitHub API error: {e}"}
    except Exception as e:
        return {"error": f"Failed to parse GitHub repo: {e}"}

def extract_resume_text(file_path: str) -> str:
    """Extracts relevant text from a resume PDF, focusing on projects and skills."""
    try:
        if not file_path or not isinstance(file_path, str):
            return "Invalid resume file path provided."
        if not os.path.exists(file_path):
            return f"Resume file not found at: {file_path}."
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        project_section = re.search(r"(?i)(?:projects|experience|work)[\s\S]*?(?=(?:education|skills|$))", text)
        skills_section = re.search(r"(?i)skills[\s\S]*?(?=(?:education|projects|$))", text)
        extracted = ""
        if project_section:
            extracted += f"Projects: {project_section.group(0).strip()}\n"
        if skills_section:
            extracted += f"Skills: {skills_section.group(0).strip()}"
        return extracted if extracted else text if text else "No readable text found in resume."
    except Exception as e:
        return f"Failed to read resume PDF: {e}. Ensure the file is a valid PDF."

def rag_role_match(project_data: Dict, job_descriptions: List[str]) -> List[Dict]:
    """Matches projects to job roles using RAG."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docs = [Document(page_content=jd) for jd in job_descriptions]
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        project_summary = f"Project: {project_data.get('name', '')}\nTech: {', '.join(str(k) for k in project_data.get('languages', {}).keys())}\nDesc: {project_data.get('description', '')}\nImpact: {project_data.get('impact', '')}"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_jobs = retriever.get_relevant_documents(project_summary)
        
        return [{"job": doc.page_content[:200], "score": i/10} for i, doc in enumerate(relevant_jobs, start=8)]
    except Exception as e:
        return [{"error": f"RAG matching failed: {e}. Ensure job descriptions are valid."}]

def refine_project_description(project_data: Dict, role: str) -> Dict:
    """Generates role-specific project descriptions for resume, LinkedIn, and pitch."""
    try:
        prompt = PromptTemplate(
            input_variables=["project_name", "tech", "desc", "impact", "role"],
            template="""Craft three concise project descriptions for a {role} role:
            1. Resume (formal, 50 words)
            2. LinkedIn (engaging, 70 words)
            3. Recruiter Pitch (persuasive, 30 words)
            Project: {project_name}
            Tech: {tech}
            Description: {desc}
            Impact: {impact}"""
        )
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        result = llm.invoke(prompt.format(
            project_name=project_data.get("name", "Unknown Project"),
            tech=", ".join(str(k) for k in project_data.get("languages", {}).keys()),
            desc=project_data.get("description", "No description"),
            impact=project_data.get("impact", "No impact metrics"),
            role=role
        )).content
        resume = result.split("1. Resume")[1].split("2. LinkedIn")[0].strip() if "1. Resume" in result else result
        linkedin = result.split("2. LinkedIn")[1].split("3. Recruiter Pitch")[0].strip() if "2. LinkedIn" in result else result
        pitch = result.split("3. Recruiter Pitch")[1].strip() if "3. Recruiter Pitch" in result else result
        return {"resume": resume, "linkedin": linkedin, "pitch": pitch}
    except Exception as e:
        return {"error": f"Failed to refine description: {e}"}

def provide_alignment_feedback(project_desc: str, job_desc: str) -> str:
    """Analyzes alignment and provides feedback with domain-specific suggestions."""
    try:
        prompt = PromptTemplate(
            input_variables=["project_desc", "job_desc"],
            template="""Analyze alignment between this project description and job description. Provide specific improvements, including domain-specific keywords (e.g., Agile, CI/CD) and recruiter expectations (e.g., teamwork, communication):
            Project: {project_desc}
            Job: {job_desc}"""
        )
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        return llm.invoke(prompt.format(project_desc=project_desc, job_desc=job_desc)).content
    except Exception as e:
        return f"Failed to provide feedback: {e}"

# ----------------- ⚙ Define Tools -----------------

tools = [
    Tool(
        name="GitHubParser",
        func=parse_github_repo,
        description="Extracts project details (tech, contributions, impact) from a GitHub repo URL."
    ),
    Tool(
        name="ResumeParser",
        func=extract_resume_text,
        description="Extracts project and skills text from a resume PDF file path."
    ),
    Tool(
        name="RoleMatcher",
        func=lambda x: rag_role_match(json.loads(x), [
            "Software Engineer: Proficient in Python, Django, REST APIs. Experience in scalable web apps, Agile workflows, and CI/CD pipelines.",
            "Data Scientist: Skilled in ML, Python, TensorFlow, Pandas. Knowledge of data pipelines, statistical analysis, and visualization."
        ]),
        description="Matches project details to job roles using RAG. Input: JSON string of project data."
    ),
    Tool(
        name="DescriptionRefiner",
        func=lambda x: refine_project_description(json.loads(x.split("|")[0]), x.split("|")[1]),
        description="Refines project description for a specific role. Input format: {project_data_json}|{role}"
    ),
    Tool(
        name="FeedbackProvider",
        func=lambda x: provide_alignment_feedback(x.split("|")[0], x.split("|")[1]),
        description="Provides feedback on project-job alignment. Input format: {project_desc}|{job_desc}"
    )
]

# ----------------- 🤖 Initialize LLM -----------------

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.5,
    max_tokens=2000,
    google_api_key=GOOGLE_API_KEY
)

# ----------------- 🧠 Initialize Agent Executors -----------------

project_analysis_agent = initialize_agent(
    tools=[tools[0], tools[1]],  # GitHubParser, ResumeParser
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

role_matching_agent = initialize_agent(
    tools=[tools[2]],  # RoleMatcher
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

content_refinement_agent = initialize_agent(
    tools=[tools[3]],  # DescriptionRefiner
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

alignment_feedback_agent = initialize_agent(
    tools=[tools[4]],  # FeedbackProvider
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ----------------- 🎨 Streamlit Interface -----------------

st.set_page_config(page_title="Placement Match Maximizer", layout="wide")
st.title("Placement Match Maximizer")
st.markdown("""
This tool helps students position their projects for job applications by analyzing GitHub repositories and resumes, matching skills to job roles, refining project descriptions, and providing alignment feedback.
""")

# Input Section
st.header("Input Your Details")
github_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/yourusername/yourproject")
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Initialize session state for storing results
if "project_data" not in st.session_state:
    st.session_state.project_data = None
if "matched_roles" not in st.session_state:
    st.session_state.matched_roles = None
if "refined_desc" not in st.session_state:
    st.session_state.refined_desc = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None

# Process Inputs
if st.button("Analyze Project and Resume"):
    if not github_url or not resume_file:
        st.error("Please provide both a GitHub URL and a resume PDF.")
    else:
        with st.spinner("Analyzing project and resume..."):
            try:
                # Save uploaded PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(resume_file.read())
                    resume_path = temp_file.name
                
                # Run Project Analysis Agent
                project_data = project_analysis_agent.run(f"Parse GitHub repo: {github_url} and resume: {resume_path}")
                project_data = json.loads(project_data) if isinstance(project_data, str) else project_data
                st.session_state.project_data = project_data
                
                # Clean up temporary file
                os.unlink(resume_path)
                
                st.subheader("🔍 Project Analysis Results")
                if "error" in project_data:
                    st.error(project_data["error"])
                else:
                    st.json(project_data)
            except Exception as e:
                st.error(f"Project analysis failed: {e}")

# Role Matching
if st.session_state.project_data and st.button("Match Roles"):
    with st.spinner("Matching roles..."):
        try:
            matched_roles = role_matching_agent.run(json.dumps(st.session_state.project_data))
            st.session_state.matched_roles = matched_roles
            st.subheader("🎯 Role Matching Results")
            if isinstance(matched_roles, list) and "error" in matched_roles[0]:
                st.error(matched_roles[0]["error"])
            else:
                st.write(matched_roles)
        except Exception as e:
            st.error(f"Role matching failed: {e}")

# Content Refinement
if st.session_state.project_data and st.button("Refine Descriptions"):
    with st.spinner("Refining descriptions..."):
        try:
            role = "Software Engineer"  # Fixed role for simplicity; can be made selectable
            refined_desc = content_refinement_agent.run(f"{json.dumps(st.session_state.project_data)}|{role}")
            refined_desc = json.loads(refined_desc) if isinstance(refined_desc, str) else refined_desc
            st.session_state.refined_desc = refined_desc
            st.subheader("✍ Content Refinement Results")
            if "error" in refined_desc:
                st.error(refined_desc["error"])
            else:
                st.json(refined_desc)
        except Exception as e:
            st.error(f"Content refinement failed: {e}")

# Alignment Feedback
if st.session_state.refined_desc and st.button("Get Alignment Feedback"):
    with st.spinner("Generating feedback..."):
        try:
            job_desc = "Software Engineer: Proficient in Python, Django, REST APIs. Experience in scalable web apps, Agile workflows, and CI/CD pipelines."
            feedback = alignment_feedback_agent.run(f"{st.session_state.refined_desc.get('resume', 'No resume description')}|{job_desc}")
            st.session_state.feedback = feedback
            st.subheader("📝 Alignment Feedback")
            st.markdown(feedback)
        except Exception as e:
            st.error(f"Alignment feedback failed: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain | Placement Match Maximizer © 2025")