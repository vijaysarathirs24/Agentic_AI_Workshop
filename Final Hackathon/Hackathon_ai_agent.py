import streamlit as st
import sqlite3
import json
import time
import re
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentExecutor
import uuid
import os
import requests
from io import BytesIO
from PIL import Image



# Custom CSS for an attractive and professional UI with updated text color
st.markdown(
    """
    <style>
    html, body, .main {
        background: radial-gradient(circle at top left, #1a1a1d, #0f0f10);
        color: #F8F8F8;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header */
    .header {
        background: linear-gradient(90deg, #00ffc3, #00b3ff);
        padding: 18px;
        text-align: center;
        border-radius: 12px;
        color: #0f0f10;
        font-size: 26px;
        font-weight: 600;
        margin-bottom: 30px;
        box-shadow: 0 8px 20px rgba(0, 255, 195, 0.2);
        text-shadow: 1px 1px 1px #000;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #00ffc3, #00b3ff);
        color: #0f0f10;
        border: none;
        border-radius: 30px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px rgba(0, 255, 195, 0.3);
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 18px rgba(0, 255, 195, 0.6);
        background: linear-gradient(135deg, #00b3ff, #00ffc3);
    }

    /* Text area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        border: 2px solid #00ffc3;
        border-radius: 12px;
        padding: 14px;
        font-size: 15px;
        transition: border-color 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #00b3ff;
        outline: none;
    }

    /* Cards and sections */
    .card, .tab-content, .expander {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 195, 0.2);
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0, 255, 195, 0.08);
    }

    .card:hover {
        box-shadow: 0 12px 30px rgba(0, 255, 195, 0.2);
        transform: translateY(-3px);
    }

    /* Selectbox */
    .stSelectbox {
        background: rgba(255, 255, 255, 0.04);
        color: #F8F8F8;
        border: 2px solid #00b3ff;
        border-radius: 10px;
        padding: 8px;
    }

    /* Footer */
    .footer {
        background: #1a1a1d;
        padding: 12px;
        text-align: center;
        margin-top: 30px;
        border-radius: 8px;
        font-size: 14px;
        color: #888;
    }

    .footer a {
        color: #00ffc3;
        text-decoration: none;
        margin: 0 10px;
        transition: color 0.3s;
    }

    .footer a:hover {
        color: #00b3ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Header and Footer
st.markdown('<div class="header"> Idea Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">© 2025 xAI | <a href="https://docs.google.com/spreadsheets/d/1_IuogQ6dE0BBAY4EW14LUWem-7qnwZCmnOD2EFfNRkM/edit?usp=sharing">Contact</a> '
'| <a href="https://docs.google.com/document/d/1j63A3CIQknFSmv3bRt2nX4nhFZeIwMJph_en1XXqkYI/edit?usp=sharing">Privacy</a></div>', unsafe_allow_html=True)

# Initialize LangChain with Gemini 2.0 Flash
os.environ["GOOGLE_API_KEY"] = "AIzaSyAN11nQIl3Vw0D6bl3qBrQKdpG3Wkg9oFk"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Sample UX/Design Trends Knowledge Base
knowledge_base = [
    Document(page_content="Trend 2025: Dark mode is increasingly popular for reducing eye strain."),
    Document(page_content="Trend 2025: Minimalist UI with large touch targets improves accessibility."),
    Document(page_content="Trend 2025: Animated microinteractions enhance user engagement."),
]
vector_store = FAISS.from_documents(knowledge_base, embeddings)

# Database setup with project name counter
def init_db():
    conn = sqlite3.connect("projects.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id TEXT PRIMARY KEY, project_name TEXT, metadata TEXT, features TEXT, prototype_feedback TEXT,
                  pitch_materials TEXT, test_plan TEXT, refined_outputs TEXT)''')
    conn.commit()
    # Get the next project name number
    c.execute("SELECT COUNT(*) FROM projects")
    project_count = c.fetchone()[0] + 1
    conn.close()
    return project_count

# Agent 1: Feature & Naming Generator Agent
def feature_naming_agent(metadata):
    start_time = time.time()
    max_attempts = 3
    for attempt in range(max_attempts):
        prompt = PromptTemplate(
            input_variables=["metadata"],
            template="Based on the project metadata: {metadata}, generate EXACTLY 5 unique product names with rationales and EXACTLY 5 prioritized features aligned with user needs. Each feature must have 'name', 'priority' (1-5), 'description', 'user_need', and 'icon_idea' (include a specific idea for at least 3 features, use 'None' for others if needed). Output MUST be raw JSON with keys 'names' (list of dictionaries with 'name' and 'rationale') and 'features' (list of dictionaries with 'name', 'priority', 'description', 'user_need', and 'icon_idea'). Do not use markdown code blocks or deviate from this structure."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({"metadata": metadata})
        try:
            text = response["text"]
            json_text = re.sub(r'^```json\n|\n```$', '', text, flags=re.MULTILINE).strip()
            output = json.loads(json_text)
            if not isinstance(output.get("names", []), list) or not isinstance(output.get("features", []), list):
                raise json.JSONDecodeError("Invalid structure", json_text, 0)
            for f in output.get("features", []):
                if not all(k in f for k in ["name", "priority", "description", "user_need", "icon_idea"]):
                    raise json.JSONDecodeError("Missing required keys in features", json_text, 0)
        except (json.JSONDecodeError, KeyError) as e:
            output = {"error": f"Invalid JSON response (Attempt {attempt + 1}/{max_attempts}): {str(e)}", "raw_response": response.get("text", response)}
            if attempt == max_attempts - 1:
                return output, time.time() - start_time
            continue
        
        if "error" not in output and (len(output.get("names", [])) != 5 or len(output.get("features", [])) != 5):
            output = {"error": f"Feature & Naming Agent must return exactly 5 names and 5 features (Attempt {attempt + 1}/{max_attempts}). Received: {len(output.get('names', []))} names, {len(output.get('features', []))} features", "raw_response": response.get("text", response)}
            if attempt == max_attempts - 1:
                return output, time.time() - start_time
            continue
        if "error" not in output and sum(1 for f in output.get("features", []) if f.get("icon_idea") is not None) < 3:
            output = {"error": f"Feature & Naming Agent must include icon ideas for at least 3 features (Attempt {attempt + 1}/{max_attempts}). Received: {sum(1 for f in output.get('features', []) if f.get('icon_idea') is not None)}", "raw_response": response.get("text", response)}
            if attempt == max_attempts - 1:
                return output, time.time() - start_time
            continue
        
        return output, time.time() - start_time
    
    raise ValueError("Feature & Naming Agent failed after maximum attempts")

# Agent 2: Prototype Refiner Agent
def prototype_refiner_agent(prototype_details):
    start_time = time.time()
    prompt = PromptTemplate(
        input_variables=["prototype_details"],
        template="Evaluate the prototype details: {prototype_details} against UX/UI standards (e.g., accessibility, interaction). Identify issues and suggest EXACTLY 3 improvements (e.g., reduce taps, icon-based navigation). Output MUST be raw JSON with key 'improvements' (list of strings). Do not use markdown code blocks."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"prototype_details": prototype_details})
    try:
        text = response["text"]
        json_text = re.sub(r'^```json\n|\n```$', '', text, flags=re.MULTILINE).strip()
        output = json.loads(json_text)
        if not isinstance(output.get("improvements", []), list):
            raise json.JSONDecodeError("Invalid structure", json_text, 0)
    except (json.JSONDecodeError, KeyError) as e:
        output = {"error": f"Invalid JSON response: {str(e)}", "raw_response": response.get("text", response)}
    
    if "error" not in output and len(output.get("improvements", [])) != 3:
        raise ValueError(f"Prototype Refiner Agent must suggest exactly 3 improvements. Received: {len(output.get('improvements', []))}")    
    
    return output, time.time() - start_time

# Agent 3: Pitch Deck Composer Agent
def pitch_deck_composer_agent(metadata, features, prototype_feedback):
    start_time = time.time()
    prompt = PromptTemplate(
        input_variables=["metadata", "features", "prototype_feedback"],
        template="Using project metadata: {metadata}, features: {features}, and prototype feedback: {prototype_feedback}, create a pitch deck with EXACTLY 6 slides (Problem & Impact, Target Audience, Product Overview, Execution Plan, Business Model, Demo Screens) and a 3-minute script (<500 words) with timing cues. Output MUST be raw JSON with keys 'slides' (list of 6 strings) and 'script' (string). Do not use markdown code blocks."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"metadata": metadata, "features": features, "prototype_feedback": prototype_feedback})
    try:
        text = response["text"]
        json_text = re.sub(r'^```json\n|\n```$', '', text, flags=re.MULTILINE).strip()
        output = json.loads(json_text)
        if not isinstance(output.get("slides", []), list) or not isinstance(output.get("script", ""), str):
            raise json.JSONDecodeError("Invalid structure", json_text, 0)
    except (json.JSONDecodeError, KeyError) as e:
        output = {"error": f"Invalid JSON response: {str(e)}", "raw_response": response.get("text", response)}
    
    if "error" not in output and (len(output.get("slides", [])) != 6 or len(output.get("script", "").split()) > 500):
        raise ValueError(f"Pitch Deck Composer Agent failed to meet slide or script requirements. Received: {len(output.get('slides', []))} slides, {len(output.get('script', '').split())} words")
    
    return output, time.time() - start_time

# Agent 4: User Testing Strategy Agent
def user_testing_strategy_agent(metadata, features, prototype_details):
    start_time = time.time()
    prompt = PromptTemplate(
        input_variables=["metadata", "features", "prototype_details"],
        template="Based on metadata: {metadata}, features: {features}, and prototype details: {prototype_details}, design a user testing plan with EXACTLY 3 objectives, EXACTLY 5 diverse testers, a delivery method, and a script. Output MUST be raw JSON with keys 'objectives' (list of 3 strings), 'testers' (list of 5 strings), 'delivery_method' (string), and 'script' (string). Do not use markdown code blocks."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"metadata": metadata, "features": features, "prototype_details": prototype_details})
    try:
        text = response["text"]
        json_text = re.sub(r'^```json\n|\n```$', '', text, flags=re.MULTILINE).strip()
        output = json.loads(json_text)
        if not all(k in output for k in ["objectives", "testers", "delivery_method", "script"]):
            raise json.JSONDecodeError("Invalid structure", json_text, 0)
    except (json.JSONDecodeError, KeyError) as e:
        output = {"error": f"Invalid JSON response: {str(e)}", "raw_response": response.get("text", response)}
    
    if "error" not in output and (len(output.get("objectives", [])) != 3 or len(output.get("testers", [])) != 5):
        raise ValueError(f"User Testing Strategy Agent failed to meet minimum requirements. Received: {len(output.get('objectives', []))} objectives, {len(output.get('testers', []))} testers")
    
    return output, time.time() - start_time

# Agent 5: Feedback Integration & Refinement Agent with RAG
def feedback_integration_agent(feedback, prior_outputs):
    start_time = time.time()
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(feedback)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = PromptTemplate(
        input_variables=["feedback", "prior_outputs", "context"],
        template="Using feedback: {feedback}, prior outputs: {prior_outputs}, and context from UX/design trends: {context}, summarize feedback, and propose EXACTLY 3 specific updates to pitch or prototype. Output MUST be raw JSON with keys 'summary' (string), 'recommendations' (list of 3 strings), and 'updated_materials' (string). Do not use markdown code blocks."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"feedback": feedback, "prior_outputs": prior_outputs, "context": context})
    try:
        text = response["text"]
        json_text = re.sub(r'^```json\n|\n```$', '', text, flags=re.MULTILINE).strip()
        output = json.loads(json_text)
        if not all(k in output for k in ["summary", "recommendations", "updated_materials"]):
            raise json.JSONDecodeError("Invalid structure", json_text, 0)
    except (json.JSONDecodeError, KeyError) as e:
        output = {"error": f"Invalid JSON response: {str(e)}", "raw_response": response.get("text", response)}
    
    if "error" not in output and len(output.get("recommendations", [])) != 3:
        raise ValueError(f"Feedback Integration Agent failed to suggest exactly 3 recommendations. Received: {len(output.get('recommendations', []))}")    
    
    return output, time.time() - start_time

# Streamlit App
def main():
    # Sidebar with logo and navigation
    st.sidebar.image("https://cdn.dribbble.com/userupload/41572065/file/original-790368ad07334cb1083c82d4f66b8069.jpg?format=webp&resize=400x300&vertical=center", width=150)
    st.sidebar.title("IKIGAI Foundation")
    st.sidebar.markdown("### Welcome to Idea Generation!")
    page = st.sidebar.radio("Navigate", ["Workflow", "Dashboard"], index=0, label_visibility="collapsed")

    # Initialize database and get next project number
    project_count = init_db()

    if page == "Workflow":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header(" Project Idea Workflow")
        st.text("Respond to the Below Questions to execute the Workflow:")
        
        with st.form(key="project_form", clear_on_submit=True):
            metadata = st.text_area(" Enter Your Project Idea (e.g., purpose, target users)", height=150, placeholder="e.g., A wellness app for remote workers aged 25-45")
            prototype_details = st.text_area(" Enter Prototype Details (e.g., screen flow, features)", height=150, placeholder="e.g., Simple UI with breathing exercises")
            feedback = st.text_area(" Enter Feedback (optional)", height=100, placeholder="e.g., Users want more customization")
            submit_button = st.form_submit_button(label=" Generate Ideas")

        if submit_button:
            if not metadata or not prototype_details:
                st.error("Please fill in both the Project Idea and Prototype Details fields.")
            else:
                project_id = str(uuid.uuid4())
                project_name = f"Project {project_count}"
                with st.spinner("🔄 Generating your ideas..."):
                    # Run all agents
                    features, feature_time = feature_naming_agent(metadata)
                    proto_feedback, proto_time = prototype_refiner_agent(prototype_details)
                    pitch_materials, pitch_time = pitch_deck_composer_agent(metadata, json.dumps(features), json.dumps(proto_feedback))
                    test_plan, test_time = user_testing_strategy_agent(metadata, json.dumps(features), prototype_details)
                    refined_outputs = {}
                    if feedback:
                        refined_outputs, feedback_time = feedback_integration_agent(feedback, json.dumps([features, proto_feedback, pitch_materials, test_plan]))
                    else:
                        # Initialize with default empty JSON if no feedback
                        refined_outputs = {"summary": "", "recommendations": [], "updated_materials": ""}

                    # Display results in cards
                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader(" Generated Features & Names")
                        st.write(f"**Project Name:** {project_name}")
                        st.write("**Product Names:**")
                        for name in features.get("names", []):
                            st.write(f"- {name['name']}: {name['rationale']}")
                        st.write("**Prioritized Features:**")
                        for feature in features.get("features", []):
                            icon = feature.get("icon_idea", "None")
                            st.write(f"- {feature['name']} (Priority {feature['priority']}): {feature['description']} (Need: {feature['user_need']}, Icon: {icon})")
                        st.write(f"⏱ Processed in {feature_time:.2f} seconds")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader(" Prototype Feedback")
                        st.write("**Improvements:**")
                        for imp in proto_feedback.get("improvements", []):
                            st.write(f"- {imp}")
                        st.write(f"⏱ Processed in {proto_time:.2f} seconds")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader(" Pitch Deck & Script")
                        st.write("**Slides:**")
                        for i, slide in enumerate(pitch_materials.get("slides", []), 1):
                            st.write(f"{i}. {slide}")
                        st.write("**Script:**")
                        st.write(pitch_materials.get("script", ""))
                        st.write(f"⏱ Processed in {pitch_time:.2f} seconds")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader(" User Testing Plan")
                        st.write("**Objectives:**")
                        for obj in test_plan.get("objectives", []):
                            st.write(f"- {obj}")
                        st.write("**Testers:**")
                        for tester in test_plan.get("testers", []):
                            st.write(f"- {tester}")
                        st.write(f"**Delivery Method:** {test_plan.get('delivery_method', '')}")
                        st.write("**Script:**")
                        st.write(test_plan.get("script", ""))
                        st.write(f"⏱ Processed in {test_time:.2f} seconds")
                        st.markdown('</div>', unsafe_allow_html=True)

                    if feedback:
                        with st.container():
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.subheader(" Refined Outputs")
                            st.write("**Summary:**")
                            st.write(refined_outputs.get("summary", ""))
                            st.write("**Recommendations:**")
                            for rec in refined_outputs.get("recommendations", []):
                                st.write(f"- {rec}")
                            st.write("**Updated Materials:**")
                            st.write(refined_outputs.get("updated_materials", ""))
                            st.write(f"⏱ Processed in {feedback_time:.2f} seconds")
                            st.markdown('</div>', unsafe_allow_html=True)

                    # Store in Database
                    conn = sqlite3.connect("projects.db")
                    c = conn.cursor()
                    features_data = features if "error" not in features else {"names": [], "features": []}
                    proto_feedback_data = proto_feedback if "error" not in proto_feedback else {"improvements": []}
                    pitch_materials_data = pitch_materials if "error" not in pitch_materials else {"slides": [], "script": ""}
                    test_plan_data = test_plan if "error" not in test_plan else {"objectives": [], "testers": [], "delivery_method": "", "script": ""}
                    refined_outputs_data = refined_outputs if "error" not in refined_outputs else {"summary": "", "recommendations": [], "updated_materials": ""}
                    c.execute('''INSERT INTO projects (id, project_name, metadata, features, prototype_feedback, pitch_materials, test_plan, refined_outputs)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                              (project_id, project_name, metadata, json.dumps(features_data), json.dumps(proto_feedback_data),
                               json.dumps(pitch_materials_data), json.dumps(test_plan_data), json.dumps(refined_outputs_data)))
                    conn.commit()
                    conn.close()

                    st.success(" Workflow completed and data stored!")

        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Dashboard":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header(" History of Projects")

        conn = sqlite3.connect("projects.db")
        c = conn.cursor()
        c.execute("SELECT id, project_name, metadata, features, prototype_feedback, pitch_materials, test_plan, refined_outputs FROM projects")
        projects = c.fetchall()
        conn.close()

        if projects:
            selected_project = st.selectbox(" Select Project", [p[1] or "Unnamed Project" for p in projects], index=0)
            if selected_project:
                conn = sqlite3.connect("projects.db")
                c = conn.cursor()
                c.execute("SELECT * FROM projects WHERE project_name = ?", (selected_project,))
                project = c.fetchone()
                conn.close()

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f" Project Details for {project[1] if project[1] else 'Unnamed Project'}")
                st.write("** Metadata:**", project[2])
                features_data = json.loads(project[3] or '{}')
                if "features" in features_data and features_data["features"]:
                    st.write("** Features:**")
                    for feature in features_data["features"]:
                        icon = feature.get("icon_idea", "None")
                        st.write(f"- {feature['name']} (Priority {feature['priority']}): {feature['description']} (Need: {feature['user_need']}, Icon: {icon})")
                else:
                    st.write("** Features:** No valid feature data available.")
                st.write("** Prototype Feedback:**")
                proto_feedback_data = json.loads(project[4] or '{}')
                for imp in proto_feedback_data.get("improvements", []):
                    st.write(f"- {imp}")
                st.write("** Pitch Materials:**")
                pitch_materials_data = json.loads(project[5] or '{}')
                for i, slide in enumerate(pitch_materials_data.get("slides", []), 1):
                    st.write(f"{i}. {slide}")
                st.write("** Script:**", pitch_materials_data.get("script", ""))
                st.write("** Test Plan:**")
                test_plan_data = json.loads(project[6] or '{}')
                for obj in test_plan_data.get("objectives", []):
                    st.write(f"- {obj}")
                st.write("**👥 Testers:**")
                for tester in test_plan_data.get("testers", []):
                    st.write(f"- {tester}")
                st.write(f"** Delivery Method:** {test_plan_data.get('delivery_method', '')}")
                st.write("** Script:**", test_plan_data.get("script", ""))
                st.write("** Refined Outputs:**")
                refined_outputs_data = json.loads(project[7] or '{}') if project[7] else {"summary": "", "recommendations": [], "updated_materials": ""}
                st.write("** Summary:**", refined_outputs_data.get("summary", ""))
                st.write("** Recommendations:**")
                for rec in refined_outputs_data.get("recommendations", []):
                    st.write(f"- {rec}")
                st.write("** Updated Materials:**", refined_outputs_data.get("updated_materials", ""))
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write("🚫 No projects available. Start by creating one in the Workflow tab!")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()