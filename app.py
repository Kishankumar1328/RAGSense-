Oimport os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Apply Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
PDF_STORAGE_PATH = "document_store/pdfs/"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)  # Ensure the directory exists

EMBEDDING_MODEL = OllamaEmbeddings(model="llama3:8b")
DOCUMENT_VECTOR_DB = Chroma(embedding_function=EMBEDDING_MODEL)
LANGUAGE_MODEL = Ollama(model="llama3:8b")

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Function to Save Uploaded File
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Function to Load PDF Documents
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

# Function to Chunk Documents
def chunk_documents(documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_processor.split_documents(documents)

# Function to Index Documents in ChromaDB
def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# Function to Search for Relevant Documents
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Function to Generate AI Response
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Streamlit UI
st.title("📘RAGSense")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant"):
            st.write(ai_response)
