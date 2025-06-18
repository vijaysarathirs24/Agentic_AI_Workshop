🎯 Placement Match Maximizer
A smart AI-powered assistant to help students and freshers align their resume, projects, and GitHub work with relevant 2025 job roles.

🚀 Overview
Placement Match Maximizer takes in your resume (PDF/TXT), project write-up, or GitHub repository link and performs the following tasks:

🔍 Project Analysis – Extracts technologies used, contributions made, and real-world impact.

🧠 Role Matching – Matches your skills to suitable 2025 entry-level job roles using Retrieval-Augmented Generation (RAG) and Tavily search.

✍️ Content Refinement – Rewrites your resume bullets, LinkedIn summary, recruiter pitch, and even a 280-character post!

📌 Company Suggestions – Recommends 3–5 companies hiring freshers for matched roles and explains why.

🧑‍🏫 Alignment Feedback – Offers feedback on articulation gaps and suggestions for better alignment.

📚 Skill Enhancement – Suggests relevant online courses/certifications to fill skill gaps and boost job readiness.

🧠 Tech Stack
Component	Technology
LLM	Gemini 1.5 Flash (google-generative-ai)
Framework	Streamlit
RAG	LangChain + FAISS + Google Embeddings
Search Tool	Tavily API
PDF Extraction	pdfplumber
Version Control	Git + GitHub

📦 Installation
1. Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/placement-match-maximizer.git
cd placement-match-maximizer
2. Create a virtual environment:
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
3. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Sample requirements.txt:

txt
Copy
Edit
streamlit
requests
pdfplumber
langchain
langchain-community
langchain-google-genai
faiss-cpu
4. Set environment variables
You can create a .env file or export these in your terminal:

bash
Copy
Edit
export GOOGLE_API_KEY=your_google_api_key
export TAVILY_API_KEY=your_tavily_api_key
Or set in Python before using them (already done in the script).

🛠️ How to Run
bash
Copy
Edit
streamlit run app.py
🖼️ Features Walkthrough
📝 Input Types
Upload a Resume (PDF/TXT)

Paste a Project Writeup

Provide a GitHub Link

📈 Step-by-Step Analysis:
Project Analysis: Extracts technologies, contributions, and impact.

Role Matching: Maps to 2025 tech job roles using Tavily + FAISS fallback.

Content Refinement: Resume lines, recruiter pitch, LinkedIn blurb, and target companies.

Feedback & Skill Suggestions: Articulation feedback and course recommendations.

📌 Example Output
vbnet
Copy
Edit
Technologies: Python, Streamlit, LangChain
Contributions: Built a RAG-based placement analyzer with Gemini AI
Impact: Helped 100+ students improve project articulation and job alignment

Best Roles: Data Analyst, AI Engineer, Software Developer
Target Companies: TCS, Infosys, Zoho, Wipro, Google

Resume Description:
- Developed a GenAI-powered placement tool using LangChain & Gemini, improving project-job alignment accuracy by 60%.

LinkedIn Post:
Built an AI tool to match my projects with 2025 job roles using Gemini & LangChain! 💼🔥 #GenAI #Placements
🔐 API Keys Required
Google Generative AI (Gemini) – https://makersuite.google.com/app/apikey

Tavily Search – https://app.tavily.com
