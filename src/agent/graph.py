from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
import os
import uuid
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph
from IPython.display import display, Image
import streamlit as st

load_dotenv()
os.environ["OPENWEATHERMAP_API_KEY"] = "8f9abeae1383f9be2f00affea4ed6501"
openai_api_key = os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    city: str


weather = OpenWeatherMapAPIWrapper()
llm = ChatOllama(model="llama3.2")
# **Node 1: Extract city from user input**


def agent(state):
    # Extract the latest user message
    user_input = state["messages"][-1].content

    res = llm.invoke(f"""
    You are given one question and you have to extract the city name from it.
    Respond ONLY with the city name. If you cannot find a city, respond with an empty string.

    Here is the question:
    {user_input}
    """)

    city_name = res.content.strip()
    if not city_name:
        return {"messages": [AIMessage(content="I couldn't find a city name in your question.")]}

    return {"messages": [AIMessage(content=f"Extracted city: {city_name}")], "city": city_name}


# **Node 2: Fetch weather information**
def weather_tool(state):
    city_name = state.get("city", "").strip()  # Retrieve city name from state

    if not city_name:
        return {"messages": [AIMessage(content="No city name provided. Cannot fetch weather.")]}

    weather_info = weather.run(city_name)
    return {"messages": [AIMessage(content=weather_info)]}


# **Define the State**
class State(TypedDict):
    messages: Annotated[list, add_messages]
    city: str  # Adding 'city' key to track extracted city name


# **Setup Workflow**
memory = MemorySaver()
workflow = StateGraph(State)

# **Define Transitions Between Nodes**
workflow.add_edge(START, "agent")
# **Add Nodes**
workflow.add_node("agent", agent)
workflow.add_node("weather", weather_tool)

# **Connect Nodes**
workflow.add_edge("agent", "weather")
workflow.add_edge("weather", END)

# **Compile Workflow with Memory Checkpointer**
app = workflow.compile(checkpointer=memory)

# === Streamlit UI === #
st.set_page_config(page_title="Weather Agent Chatbot", layout="centered")
st.title("🌦️ Weather Agent (OpenAI-powered)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])    

# Chat Input Handling
if prompt := st.chat_input("Ask about the weather (e.g., What is the weather in Tokyo?)"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        result = response["messages"][-1].content

        response_container.markdown(result)
        st.session_state.chat_history.append({"role": "assistant", "content": result})
        
graph = (app)