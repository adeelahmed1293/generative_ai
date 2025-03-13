import streamlit as st
import os
from pipeline import RAGSystem

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 InsightFlow ")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG system instance (only once)
@st.cache_resource
def load_rag_system():
    return RAGSystem()

# Load the RAG system
with st.spinner("Loading the RAG system... This may take a minute."):
    rag = load_rag_system()

# Function to check if query is conversational
def is_conversational(query):
    conversational_phrases = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "how are you", "what's up", "nice to meet you", 
        "thanks", "thank you", "bye", "goodbye"
    ]
    
    query_lower = query.lower()
    for phrase in conversational_phrases:
        if phrase in query_lower:
            return True
    
    # Check if query is too short (likely conversational)
    if len(query.split()) < 3:
        return True
        
    return False

# Function to handle conversational queries
def get_conversational_response(query):
    query_lower = query.lower()
    
    if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "greetings"]):
        return "Hello! How can I help you today?"
    
    elif any(time in query_lower for time in ["good morning", "good afternoon", "good evening"]):
        return f"{query.capitalize()}! How can I assist you?"
    
    elif "how are you" in query_lower:
        return "I'm doing well, thanks for asking! How can I help you?"
    
    elif "thank" in query_lower:
        return "You're welcome! Feel free to ask if you have more questions."
    
    elif any(farewell in query_lower for farewell in ["bye", "goodbye"]):
        return "Goodbye! Feel free to return if you have more questions."
    
    else:
        return "I'm here to help with your questions about the documents I've analyzed. What would you like to know?"

# Function to get response based on query type
def get_response(query):
    if is_conversational(query):
        return get_conversational_response(query)
    else:
        try:
            return rag.answer_question(query)
        except Exception as e:
            return f"Error: {str(e)}"

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.write("This chatbot uses a Retrieval-Augmented Generation (RAG) system to provide accurate answers based on the loaded documents.")
    
    st.header("Data Sources")
    st.write("The system retrieves information from:")
    st.write("- CSV data")
    st.write("- JSON data")
    st.write("- PDF documents")
    st.write("- Web content")
    
    # Optional: Add reset button
    # In the sidebar section, change:
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})\
    
