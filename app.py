import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NATA AI", 
    page_icon="⚡", 
    layout="centered"
)

# --- 2. CUSTOM UI / CSS MAGIC ---
st.markdown("""
    <style>
    @keyframes rgb-glow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Target the Streamlit Chat Input container */
    [data-testid="stChatInput"] {
        background: linear-gradient(90deg, #ff0080, #ff8c00, #40e0d0, #8a2be2, #ff0080);
        background-size: 200% auto;
        animation: rgb-glow 3s linear infinite;
        border-radius: 12px;
        padding: 2px;
    }
    
    /* Make the inner input text box dark so the glowing border pops */
    [data-testid="stChatInput"] > div {
        background-color: #0e1117 !important; 
        border-radius: 10px !important;
        border: none !important;
    }

    /* GEMINI-STYLE RGB TEXT FADE */
    .rgb-text {
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570, #f4b400, #4285f4);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        animation: rgb-glow 3s linear infinite;
        font-size: 2.2em;
        font-weight: 800;
        display: inline-block;
    }

    /* PURE CSS TYPEWRITER FIX (Bypasses Streamlit's JS Block!) */
    .mask-container {
        position: relative;
        display: inline-block;
        margin-bottom: 20px;
        padding-right: 5px; /* Room for the cursor */
    }
    
    /* This creates a dark box that slides over the text to emulate typing/deleting */
    .mask {
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117; /* Streamlit dark mode background color */
        border-left: 3px solid white; /* The Blinking Cursor! */
        animation: slide-mask 3.5s steps(30, end) infinite alternate;
    }
    
    @keyframes slide-mask {
        0% { width: 100%; }
        100% { width: 0%; }
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

# --- INITIALIZE SESSION STATE EARLY SO WE CAN CHECK IT FOR THE UI ---
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. LLM & PROMPT SETUP ---
llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model='llama-3.3-70b-versatile'
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly intelligent AI assistant named NATA. Remember previous conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# --- 4. MEMORY STORE ---
if "store" not in st.session_state:
    st.session_state.store = {}

store = st.session_state.store

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- 5. UI: SIDEBAR ---
with st.sidebar:
    st.title("⚡ NATA AI")
    st.caption("Powered by Groq & LangChain")
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        store.clear()
        st.rerun() 

# --- 6. UI: MAIN CHAT AREA WITH DYNAMIC HEADER ---

if len(st.session_state.messages) == 0:
    # IF NO MESSAGES: Play the Pure CSS Typewriter Loop!
    typewriter_html = """
    <div class="mask-container">
        <span class="rgb-text">Heyy NATA, Can you Helpp me...</span>
        <div class="mask"></div>
    </div>
    """
    st.markdown(typewriter_html, unsafe_allow_html=True)
    st.markdown("Welcome! I am NATA, your custom AI. What's on your mind?")
else:
    # IF MESSAGES EXIST: Lock in the static RGB Title!
    st.markdown('<div class="rgb-text" style="margin-bottom: 20px;">NATA AI</div>', unsafe_allow_html=True)

# Render chat history with custom avatars
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# --- 7. CHAT INPUT & RESPONSE ---
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = chain_with_memory.invoke(
                {'input': user_input},
                config={'configurable': {'session_id': st.session_state.session_id}}
            )
            bot_reply = response.content
            st.write(bot_reply)
    
    st.session_state.messages.append({'role': 'assistant', 'content': bot_reply})
    st.rerun()