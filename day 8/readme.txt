🚀 Placement Match Maximizer
An AI-powered assistant to help students align their GitHub projects and resumes with targeted job roles. This tool uses LangChain agents powered by Google's Gemini 2.0 Flash LLM to analyze, match, refine, and optimize project representation for job applications.

🔧 Features
GitHub Project Analyzer: Parses a GitHub repo to extract tech stack, contributions, and impact.

Resume Parser: Extracts key content (projects, skills) from uploaded resume PDFs.

RAG-based Role Matcher: Uses Retrieval-Augmented Generation (RAG) to match project data to suitable job roles.

Project Description Refiner: Crafts formal, LinkedIn-style, and recruiter-pitch versions of your project for a specific role.

Alignment Feedback Generator: Compares your project description with a job description and gives actionable improvement suggestions.

🤖 Modular AI Agent Architecture
The tool is powered by LangChain Agents, each handling a specific task independently:

Agent	Purpose
Project Analysis Agent	Combines GitHub and Resume parsers to extract and unify candidate project data
Role Matching Agent	Matches extracted project data to predefined job roles using RAG
Content Refinement Agent	Refines project descriptions based on a selected role (e.g., Software Engineer)
Alignment Feedback Agent	Provides improvement feedback by comparing refined project content to job expectations

📦 Requirements
Python 3.9+

Streamlit

LangChain

HuggingFace Transformers

FAISS

PyPDF2

GitHub API (via PyGithub)

Gemini API (via langchain-google-genai)

Install dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Note: Ensure you have access to Gemini 2.0 API and GitHub tokens in a .env file.

🔐 Environment Setup
Create a .env file in your root directory with:

env
Copy
Edit
GOOGLE_API_KEY=your_google_gemini_api_key
GITHUB_TOKEN=your_github_personal_access_token
🚀 How to Run
Run the Streamlit app locally:

bash
Copy
Edit
streamlit run app.py
🖼️ User Interface
Input GitHub repo URL and upload your resume (PDF).

Click "Analyze Project and Resume".

Click "Match Roles" to see which jobs your project aligns with.

Click "Refine Descriptions" to get:

Resume version

LinkedIn version

Recruiter pitch

Click "Get Alignment Feedback" to improve the description for a target job.

📂 Folder Structure
bash
Copy
Edit
.
├── app.py               # Main Streamlit application
├── .env                 # API keys (not committed to GitHub)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
📌 Example Job Roles
Currently supports these built-in job roles (you can modify or expand them):

Software Engineer

Data Scientist

🛠️ Tech Stack
LangChain Agents

Google Gemini 2.0 Flash

FAISS Vector Store

Streamlit

GitHub API

PyPDF2

PromptTemplates for smart prompting
