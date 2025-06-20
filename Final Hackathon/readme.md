Idea Generator – Powered by LangChain + Gemini Welcome to the Idea Generator, an AI-powered product ideation assistant built using Streamlit, LangChain, and Gemini 1.5 Flash. It helps teams turn rough project ideas into structured feature lists, refined prototypes, pitch decks, testing strategies, and more — all in minutes.

Features Feature Description

Project Metadata Input Capture your basic idea (purpose, target audience, etc.) AI-Powered Agents 5 agents analyze and generate names, features, prototypes, pitch decks, and test plans Feedback Integration Accepts feedback and recommends improvements using RAG (Retrieval-Augmented Generation) Pitch Deck Generator Generates a 6-slide pitch with a 3-minute script User Testing Planner Designs a test plan with objectives, testers, and scripts Project Dashboard Stores and retrieves past projects for reference Professional UI Attractive Streamlit theme with dark mode & modern gradients SQLite Database Saves all project sessions for future review

Tech Stack: LangChain (agents, tools, prompts) Google Gemini 1.5 Flash (via langchain-google-genai) FAISS Vector Store (UX knowledge RAG) SQLite (project storage) Streamlit (frontend) PIL (image handling)

Install dependencies: pip install -r requirements.txt

Set up Gemini API key: Edit the Python file and replace os.environ["give me your api"] with your Gemini API key or store it in an .env file and load it via python-dotenv.

Run the app: streamlit run idea_generator.py How It Works (Agent Workflow)

Feature & Naming Agent Input: Project metadata
Output: 5 unique product names + 5 prioritized features (with icons & user needs)

Validates: JSON format, required fields, icon ideas for ≥ 3 features

Prototype Refiner Agent Input: Prototype details
Output: 3 UX improvements

Validates: JSON and exact number of improvements

Pitch Deck Composer Agent Input: metadata + features + feedback
Output: 6 slides and a 3-min script

Validates: Slide count (6) and script word count (≤ 500)

User Testing Strategy Agent Input: metadata + features + prototype
Output: Objectives, 5 testers, delivery method, script

Validates: Completeness and strict field structure

Feedback Integration Agent (RAG) Input: Feedback + previous outputs
Retrieves: Related UX trends from a FAISS vector store

Output: Feedback summary, 3 update recommendations, updated materials

Data Storage All project sessions are stored in projects.db using SQLite, with each project having:

project_name, metadata, features, prototype_feedback, pitch_materials, test_plan, and refined_outputs.

UI Design Dark Theme: #617AA8 (primary), #F3F0F0 (text), #35363B (secondary)

Custom CSS: Smooth buttons, gradient headers, blurred glass cards, and polished footer

Sidebar Navigation: Workflow vs. Dashboard views

Dashboard View Users can revisit previous projects:

Browse past metadata, features, feedback, pitch decks, and testing plans

See AI-generated recommendations and scripts

Environment Variables Variable Description GOOGLE_API_KEY Gemini 1.5 API Key

Example Input/Output Input:

Project Idea: "A mindfulness app for college students" Prototype: "3-screen flow with mood tracker, audio guide, breathing tool" Output (Summary): 5 product names: MindNest, ZenBuddy, etc.

5 features: Mood journal, Breathing animation, Audio coach, etc.

3 prototype UX fixes

6-slide pitch + script

3 user testing objectives, 5 testers

Updated version after feedback
