import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import re

# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DeepSeek Code Companion")
st.caption("Developed by Siddhartha Pathak")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
  
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    st.markdown("Developed by [Siddhartha Pathak](https://www.linkedin.com/in/siddharthapathak/) | [Web](https://siddharthapathak.com.np/)")

# Initialize LLM
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# Create the system message template
system_template = """You are an expert AI coding assistant. Provide direct responses without any meta-commentary, thinking process, or self-description.
Focus only on the actual content the user needs. Always respond in English.

Current conversation:
{chat_history}

User: {user_input}
Assistant:"""

chat_prompt = ChatPromptTemplate.from_template(system_template)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! How can I help you today?"}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your question here...")

def clean_response(response):
    """Remove thinking tags and other meta-commentary from the response."""
    # Remove <think> or <thinking> tags and their content
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
    
    # Remove any other meta-commentary patterns (can be expanded based on needs)
    response = re.sub(r'\*thinking\*.*?\n', '', response, flags=re.DOTALL)
    response = re.sub(r'\[thinking\].*?\[/thinking\]', '', response, flags=re.DOTALL)
    
    # Clean up any extra whitespace
    response = re.sub(r'\n\s*\n', '\n\n', response)
    response = response.strip()
    
    return response

def generate_ai_response(user_input, chat_history):
    try:
        formatted_prompt = chat_prompt.format(
            chat_history=chat_history,
            user_input=user_input
        )
        response = llm_engine.invoke(formatted_prompt)
        cleaned_response = clean_response(response.content)
        return cleaned_response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def update_chat_history():
    chat_history = ""
    for message in st.session_state.message_log:
        role = "Assistant" if message["role"] == "ai" else "User"
        chat_history += f"{role}: {message['content']}\n\n"
    return chat_history

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Update chat history
    st.session_state.chat_history = update_chat_history()
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        ai_response = generate_ai_response(user_query, st.session_state.chat_history)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()