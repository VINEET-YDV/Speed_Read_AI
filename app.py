import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file at the very beginning
load_dotenv() 

from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration & API Key Check ---
# Perform this check right after loading .env and before running the app logic
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please add it to your .env file or set it as an environment variable.")
    st.stop()

# --- Functions for RAG Pipeline ---

@st.cache_resource(show_spinner="Embedding documents...")
def create_vector_db_from_uploads(uploaded_files):
    """
    Creates a vector database from uploaded files.
    Caches the result to avoid reprocessing on every interaction.
    """
    if not uploaded_files:
        return None

    # Use a temporary directory to safely store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # --- FIX: Simplified DirectoryLoader initialization ---
        # The modern DirectoryLoader automatically detects file types.
        # The 'loader_map' argument is no longer needed or supported.
        loader = DirectoryLoader(
            temp_dir,
            glob="**/*.[pP][dD][fF]", # Glob pattern to find PDF files
            show_progress=True,
            use_multithreading=True
        )
        # You can add more loaders for other file types if needed
        # For example, for .txt files:
        # loader_txt = DirectoryLoader(temp_dir, glob="**/*.txt", loader_cls=TextLoader)
        
        documents = loader.load()

    # Split documents into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings using a high-quality open-source model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use 'cuda' for GPU acceleration
    )

    # Create and return the Chroma vector store from the chunks
    db = Chroma.from_documents(chunks, embeddings)
    return db

def create_qa_chain(db):
    """
    Creates a question-answering chain using the vector database and the Groq LLM.
    """
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.

    Context: {context}
    Question: {question}
    Helpful Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the RetrievalQA chain with the specified components
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant chunks
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Streamlit App UI ---

st.set_page_config(page_title="DocQuery Pro ⚡️", layout="wide")
st.title("📄 DocQuery Pro: Ask Questions to Your Documents")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload up to 20 documents at a time."
    )

    if uploaded_files:
        if st.button("Process Documents"):
            # Store the created vector DB in the session state
            st.session_state.vector_db = create_vector_db_from_uploads(uploaded_files)
            if st.session_state.vector_db:
                st.success("Documents processed successfully! You can now ask questions.")
            else:
                st.error("Failed to process documents. Please ensure files are not corrupted.")
    
    st.header("Powered by")
    st.markdown("[Groq](https://groq.com/) | [LangChain](https://www.langchain.com/) | [Streamlit](https://streamlit.io/)")

# --- Main Chat Interface ---
st.header("2. Ask Your Questions")

# Only show the chat interface if documents have been processed
if 'vector_db' not in st.session_state or st.session_state.vector_db is None:
    st.warning("Please upload and process your documents in the sidebar to begin.")
else:
    # Create the QA chain once and store it in the session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = create_qa_chain(st.session_state.vector_db)

    # User input for the question
    user_question = st.text_input("What would you like to know from your documents?")

    if user_question:
        with st.spinner("Finding the answer..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": user_question})
                
                # Display the answer
                st.subheader("✨ Answer")
                st.write(result["result"])

                # Display source documents in an expander
                with st.expander("View Source Documents"):
                    for doc in result["source_documents"]:
                        st.markdown(f"**Source:** `{doc.metadata.get('source', 'N/A')}`")
                        # Display a snippet of the source content
                        st.markdown(f"**Content:**\n\n---\n\n{doc.page_content[:500]}...")
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
