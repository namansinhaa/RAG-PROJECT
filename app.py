import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Modern LangChain imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# -------------------------------------------------------------
# 1. Resource Caching (Load Once in Memory)
# -------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    # Downloads/loads the local embedding model once
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1
    )

embedding_model = get_embedding_model()
llm = get_llm()

DB_DIR = "chroma_db"

# Prompt template
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and precise AI research assistant.

Use ONLY the provided context to answer the question.
If the answer cannot be deduced from the context, state:
"I could not find the answer in the document."

Context:
{context}
"""
        ),
        (
            "human",
            "Question: {question}"
        )
    ]
)

# -------------------------------------------------------------
# 2. Session State Initialization
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    # Check if a Chroma DB already exists on disk
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_model
        )
        st.session_state.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )
    else:
        st.session_state.retriever = None

# -------------------------------------------------------------
# 3. Sidebar: Book / Document Upload & Indexing
# -------------------------------------------------------------
with st.sidebar:
    st.title("📚 Document Management")
    st.write("Upload a PDF book or document to build/update the knowledge base.")

    uploaded_file = st.file_uploader(
        "Select a PDF file",
        type=["pdf"],
        help="Upload PDF books, manuals, or research papers."
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index Document", use_container_width=True):
            with st.spinner("Processing and chunking document..."):
                # Save uploaded file temporarily to disk so PyPDFLoader can parse it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    # 1. Load PDF
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    # 2. Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)

                    # 3. Store into ChromaDB
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model,
                        persist_directory=DB_DIR
                    )

                    # 4. Update session retriever
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
                    )

                    st.success(f"Successfully processed **{uploaded_file.name}** ({len(chunks)} chunks indexed)!")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                finally:
                    # Clean up temporary disk file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------------------
# 4. Main Chat Interface
# -------------------------------------------------------------
st.title("💬 Book RAG Knowledge Assistant")
st.caption("Powered by Groq LPUs (`openai/gpt-oss-20b`) & ChromaDB")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Context"):
                for idx, src in enumerate(msg["sources"], 1):
                    page = src.metadata.get("page", "N/A")
                    st.markdown(f"**Chunk #{idx} (Page {page}):**\n> {src.page_content}")

# Handle new user query
if query := st.chat_input("Ask a question about the document..."):
    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload and process a document in the sidebar first.")
    else:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer with RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                # 1. Retrieve top-k context chunks
                retrieved_docs = st.session_state.retriever.invoke(query)
                context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # 2. Format prompt and invoke LLM
                final_prompt = PROMPT_TEMPLATE.invoke({
                    "context": context_str,
                    "question": query
                })
                response = llm.invoke(final_prompt)
                answer = response.content

                st.markdown(answer)

                # Show sources under an expander
                if retrieved_docs:
                    with st.expander("🔍 View Retrieved Context"):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Chunk #{idx} (Page {page}):**\n> {doc.page_content}")

        # Save assistant answer to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_docs
        })
        
#to run this paste this on the terminal : streamlit run app.py