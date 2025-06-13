import streamlit as st
import requests
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# ----------------------------
# Set Your API Keys
# ----------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyAN11nQIl3Vw0D6bl3qBrQKdpG3Wkg9oFk"
os.environ["TAVILY_API_KEY"] = "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"
WEATHER_API_KEY = "7986148ae0c3467f9e192828251306 "

# ----------------------------
# Weather Tool
# ----------------------------
@tool
def get_weather(city: str) -> str:
    """Get current weather information for a given city using WeatherAPI.com."""
    try:
        response = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        )
        data = response.json()
        condition = data["current"]["condition"]["text"]
        temp_c = data["current"]["temp_c"]
        return f"The current weather in {city} is {condition} with a temperature of {temp_c}°C."
    except Exception as e:
        return f"Failed to fetch weather: {e}"

# ----------------------------
# Tavily Search Tool
# ----------------------------
tavily_search = TavilySearchResults()

# ----------------------------
# Gemini 2.0 Flash LLM Setup
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # or "gemini-2.0" if you confirmed support

# ----------------------------
# Correct Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful travel assistant. Use tools to answer queries."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ----------------------------
# Create Agent + Executor
# ----------------------------
tools = [get_weather, tavily_search]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------
# Streamlit Web UI
# ----------------------------
st.set_page_config(page_title="Travel Assistant AI", layout="centered")
st.title("🌍 Travel Assistant AI")
st.markdown("Get **weather** and **top tourist attractions** for your travel destination!")

destination = st.text_input("Enter Destination City", placeholder="e.g., Tokyo")

if st.button("Get Info") and destination:
    with st.spinner("Thinking..."):
        try:
            query = f"What is the weather in {destination} and top tourist attractions?"
            response = agent_executor.invoke({"input": query})
            st.success("Here's what I found:")
            st.markdown(response["output"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")
