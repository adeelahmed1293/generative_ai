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


class RAGSystem:
    def __init__(self, csv_path="./data/FAQs.csv", json_path="./data/FAQs.json", 
                 pdf_path="./data/LLMS.pdf", web_url="https://youngreadersfoundation.org/importance-of-reading/",
                 embed_model="all-MiniLM-L6-v2", llm_model="llama-3.1-8b-instant", 
                 temperature=0, top_k=3, chunk_size=1000, chunk_overlap=200):
        """
        Initialize the RAG (Retrieval-Augmented Generation) system.
        
        Args:
            csv_path: Path to CSV data file
            json_path: Path to JSON data file
            pdf_path: Path to PDF data file
            web_url: URL for web content
            embed_model: Name of HuggingFace embedding model
            llm_model: Name of LLM model to use with Groq
            temperature: Temperature setting for LLM responses
            top_k: Number of top documents to retrieve
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between text chunks
        """
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        
        # Store configuration
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
        
        # Initialize the system
        self.load_data()
        self.process_data()
        self.create_embeddings()
        self.setup_llm()
        self.setup_retrieval_chain()
    
    def load_data(self):
        """Load data from various sources."""
        # Load CSV data
        csv_loader = CSVLoader(file_path=self.csv_path)
        self.csv_data = csv_loader.load()
        
        # Load JSON data
        json_loader = JSONLoader(
            file_path=self.json_path,
            jq_schema='map({question, answer})',
            text_content=False
        )
        self.json_data = json_loader.load()
        
        # Load PDF data
        pdf_loader = PyPDFLoader(self.pdf_path)
        self.pdf_data = pdf_loader.load()
        
        # Load web data
        web_loader = WebBaseLoader(self.web_url)
        self.web_data = web_loader.load()
    
    def process_data(self):
        """Process and split the loaded data."""
        # Combine all documents
        self.all_documents = self.json_data + self.csv_data + self.pdf_data + self.web_data
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.split_docs = text_splitter.split_documents(self.all_documents)
        
        print(f"Total chunks after splitting: {len(self.split_docs)}")
    
    def create_embeddings(self):
        """Create embeddings and vector store."""
        # Create embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(self.split_docs, self.embeddings)
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
    
    def setup_llm(self):
        """Set up the language model."""
        self.llm = ChatGroq(
            model=self.llm_model,
            temperature=self.temperature,
            api_key=self.api_key
        )
    
    def setup_retrieval_chain(self):
        """Set up the retrieval chain."""
        # Create prompt template
        prompt_template = """
                You are a helpful and knowledgeable assistant tasked with providing CONCISE answers.

                Below is the context (documents) related to the user's query.
                Use this context to answer the following question in a BRIEF and CLEAR manner.
                Focus only on the most important information and limit your response to 3-5 sentences.
                Do not start your answer with phrases like "Based on the provided context" or similar.
                Just provide the answer directly.

                Context:
                {context}

                Question: {input}

                Concise Answer:
                """
        
        # Create a PromptTemplate instance
        prompt = PromptTemplate(
            input_variables=["context", "input"], 
            template=prompt_template
        )
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create the retrieval chain
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
    
    def answer_question(self, query):
        """
        Answer a question using the RAG system.
        
        Args:
            query: User's question
            
        Returns:
            str: Concise answer to the question
        """
        # Execute the chain
        result = self.retrieval_chain.invoke({"input": query})
        
        # Extract and return only the answer
        return result['answer']
    
    def save_vector_store(self, path="./vector_store"):
        """
        Save the vector store for future use.
        
        Args:
            path: Directory path to save the vector store
        """
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")
    
    def load_vector_store(self, path="./vector_store"):
        """
        Load a previously saved vector store.
        
        Args:
            path: Directory path to load the vector store from
        """
        self.vector_store = FAISS.load_local(path, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        print(f"Vector store loaded from {path}")


# Example usage:
if __name__ == "__main__":
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Ask a question
    query = "What is Transformer?"
    answer = rag.answer_question(query)
    
    # Print the answer
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    
    # Save the vector store for future use (optional)
    # rag.save_vector_store()