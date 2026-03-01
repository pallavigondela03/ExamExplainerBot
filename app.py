import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORT SECTION ---
# Ensure these filenames exist in C:\ExamExplainerBot\
try:
    from retriever import StudyRetriever
    from chatbot_engine import ExamChatbot
    from safety_filter import SafetyFilter
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Edu-Exam Explainer Bot",
    page_icon="🎓",
    layout="wide"
)

# Initialize Backend Components 
# We use st.cache_resource so the AI model doesn't reload every time you type
@st.cache_resource
def initialize_system():
    try:
        retriever = StudyRetriever()
        chatbot = ExamChatbot()
        return retriever, chatbot
    except Exception as e:
        # This will catch if FAISS files are missing or API key is wrong
        st.error(f"❌ Initialization Error: {e}")
        return None, None

retriever, chatbot = initialize_system()

# --- UI Layout ---
st.title("🎓 Education Examination & Evaluation Explainer")
st.markdown("""
    This GenAI assistant explains academic regulations, grading patterns, and revaluation processes.
    *All answers are grounded in official institutional documents.*
""")

# Sidebar info and Status
with st.sidebar:
    st.header("System Integrity")
    if retriever and chatbot:
        st.success("✅ RAG Pipeline: Ready")
        st.success("✅ Gemini AI: Connected")
    else:
        st.error("❌ System Offline")
        if st.button("Retry Connection"):
            st.rerun()
    
    st.divider()
    st.info("""
    **Guardrails Active:**
    - No Exam Answers 🚫
    - No Grade Prediction 🚫
    - Context-Only Retrieval 📚
    """)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about grading, revaluation, or exam patterns..."):
    
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Safety Filter Layer (Static method check)
    if not SafetyFilter.is_safe(prompt):
        bot_response = "❌ **Safety Warning:** I am strictly authorized to explain academic policies. I cannot provide specific exam answers or predict your personal grades."
        context_chunks = []
    else:
        if retriever is None or chatbot is None:
            bot_response = "❌ System is not initialized. Check terminal for errors."
            context_chunks = []
        else:
            with st.spinner("Searching regulations and generating response..."):
                try:
                    # 3. Retrieval Step
                    context_chunks = retriever.get_relevant_context(prompt, top_k=3)
                    
                    # 4. Generation Step
                    if context_chunks:
                        bot_response = chatbot.generate_response(prompt, context_chunks)
                    else:
                        bot_response = "I'm sorry, I couldn't find any relevant academic regulations in my database to answer this."
                except Exception as e:
                    bot_response = f"⚠️ Processing Error: {str(e)}"
                    context_chunks = []

    # 5. Display Assistant Response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        
        # 6. Transparency Layer (Show retrieved context)
        if context_chunks:
            with st.expander("🔍 View Grounded Context (Source Transparency)"):
                for i, chunk in enumerate(context_chunks):
                    st.markdown(f"**Source {i+1}:** `{chunk['source']}`")
                    st.caption(chunk['text'])
                    st.divider()

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})