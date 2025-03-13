from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import JSONLoader, PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

class RAGSystem:
    def __init__(self, csv_path="./data/FAQs.csv", json_path="./data/FAQs.json", 
                 pdf_path="./data/LLMS.pdf", web_url="https://youngreadersfoundation.org/importance-of-reading/",
                 embed_model="all-MiniLM-L6-v2", llm_model="llama-3.1-8b-instant", 
                 temperature=0, top_k=3, chunk_size=1000, chunk_overlap=200):
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        
        self.csv_path = csv_path
        self.json_path = json_path
        self.pdf_path = pdf_path
        self.web_url = web_url
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.load_data()
        self.process_data()
        self.create_embeddings()
        self.setup_llm()
        self.setup_retrieval_chain()
    
    def load_data(self):
        csv_loader = CSVLoader(file_path=self.csv_path)
        self.csv_data = csv_loader.load()
        
        json_loader = JSONLoader(
            file_path=self.json_path,
            jq_schema='map({question, answer})',
            text_content=False
        )
        self.json_data = json_loader.load()
        
        pdf_loader = PyPDFLoader(self.pdf_path)
        self.pdf_data = pdf_loader.load()
        
        web_loader = WebBaseLoader(self.web_url)
        self.web_data = web_loader.load()
    
    def process_data(self):
        self.all_documents = self.json_data + self.csv_data + self.pdf_data + self.web_data
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.split_docs = text_splitter.split_documents(self.all_documents)
    
    def create_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.vector_store = FAISS.from_documents(self.split_docs, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
    
    def setup_llm(self):
        self.llm = ChatGroq(
            model=self.llm_model,
            temperature=self.temperature,
            api_key=self.api_key
        )
    
    def setup_retrieval_chain(self):
        prompt_template = """
            You are a helpful and knowledgeable assistant tasked with providing CONCISE answers.
            
            Context:
            {context}
            
            Question: {input}
            
            Concise Answer:
            """
        prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
    
    def answer_question(self, query):
        result = self.retrieval_chain.invoke({"input": query})
        return result['answer']
    
@st.cache_resource
def load_rag_system():
    return RAGSystem()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 InsightFlow")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("Loading the RAG system... This may take a minute."):
    rag = load_rag_system()

def is_conversational(query):
    conversational_phrases = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
                              "good evening", "how are you", "what's up", "nice to meet you", "thanks", "thank you", "bye", "goodbye"]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in conversational_phrases) or len(query.split()) < 3

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
        return "I'm here to help with your questions. What would you like to know?"

def get_response(query):
    return get_conversational_response(query) if is_conversational(query) else rag.answer_question(query)

with st.sidebar:
    st.header("About")
    st.write("This chatbot uses a Retrieval-Augmented Generation (RAG) system to provide accurate answers based on documents.")
    st.header("Data Sources")
    st.write("- CSV data\n- JSON data\n- PDF documents\n- Web content")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
