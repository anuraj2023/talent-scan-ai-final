import sys, os
sys.dont_write_bytecode = True
import io 
import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal
from PyPDF2 import PdfReader
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from llm_agent import ChatBot
from app.data_ingestor import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up file paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

FAISS_PATH = os.path.join(PARENT_DIR, "vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  1. To set up, please add your OpenAI's API key. 🔑 
  2. Type in a job description query. 💬

  Please make sure to check the sidebar for more useful information. 💡
"""

# Set up Streamlit page
st.set_page_config(page_title="Talent Scan AI")
st.title("Talent Scan AI")


# Initialize session state variables
if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "rag_selection"  not in st.session_state:
   st.session_state.rag_selection = "Generic RAG"

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []

if "rag_pipeline" not in st.session_state:
  st.session_state.retriever = None
  st.session_state.llm = None

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



def upload_files():
    """Process uploaded resume files and update the vectorstore."""
    uploaded_files = st.session_state.uploaded_files
    if uploaded_files:
        df_list = []
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    pdf_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
                    df_list.append(pd.DataFrame({"ID": [uploaded_file.name], "Resume": [pdf_text]}))
                    logger.info(f"Processed PDF file: {uploaded_file.name}")
                else:
                    st.error(f"Uploaded file {uploaded_file.name} is not a PDF.")
                    logger.warning(f"Skipped non-PDF file: {uploaded_file.name}")
            except Exception as error:
                st.error(f"Error processing file {uploaded_file.name}: {str(error)}")
                logger.error(f"Error processing file {uploaded_file.name}: {str(error)}")
        
        if df_list:
            with st.spinner('Indexing the uploaded data. This may take a while...'):
                st.session_state.df = pd.concat(df_list, ignore_index=True)
                vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
                st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
                st.session_state.retriever = st.session_state.rag_pipeline
                st.session_state.llm = ChatBot(
                  api_key=st.session_state.api_key,
                  model=st.session_state.gpt_selection,
                )
            st.success(f"Successfully processed {len(df_list)} files.")
            logger.info(f"Indexed {len(df_list)} new resume files")

def check_openai_api_key(api_key: str):
  """Verify the validity of the OpenAI API key."""
  openai.api_key = api_key
  try:
    openai.models.list()
    logger.info("OpenAI API key validated successfully")
    return True
  except openai.AuthenticationError as e:
    logger.error(f"OpenAI API key validation failed: {str(e)}")
    return False
  

def check_model_name(model_name: str, api_key: str):
  """Check if the specified model name is available in OpenAI's model list."""
  openai.api_key = api_key
  model_list = [model.id for model in openai.models.list()]
  is_valid = model_name in model_list
  if is_valid:
    logger.info(f"Validated model name: {model_name}")
  else:
    logger.warning(f"Invalid model name: {model_name}")
  return is_valid

def clear_message():
  """Clear the chat history and resume list."""
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]
  logger.info("Cleared chat history and resume list")

# Set up the main chat interface
user_query = st.chat_input("Type your message here...")

# Set up the sidebar
with st.sidebar:
  st.markdown("# Settings")

  st.text_input("OpenAI's API Key", type="password", key="api_key")
  #st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
  st.text_input("GPT Model", "gpt-3.5-turbo", key="gpt_selection")
  st.file_uploader("Upload resumes", type=["pdf"], key="uploaded_files", accept_multiple_files=True, on_change=upload_files)
  st.button("Clear conversation", on_click=clear_message)

# Display chat history
for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])

# Check for API key and model validity
if not st.session_state.api_key:
  st.info("Please add your OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_openai_api_key(st.session_state.api_key):
  st.error("The API key is incorrect. Please set a valid OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_model_name(st.session_state.gpt_selection, st.session_state.api_key):
  st.error("The model you specified does not exist. Learn more about [OpenAI models](https://platform.openai.com/docs/models).")
  st.stop()


# Process user query
if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      document_list = st.session_state.retriever.retrieve_docs(user_query, st.session_state.llm, st.session_state.rag_selection)
      query_type = st.session_state.retriever.meta_data["query_type"]
      st.session_state.resume_list = document_list
      stream_message = st.session_state.llm.generate_message_stream(user_query, document_list, [], query_type)
    end = time.time()

    response = st.write_stream(stream_message)
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, st.session_state.retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, st.session_state.retriever.meta_data, end-start))

    logger.info(f"Processed query in {end-start:.2f} seconds. Query type: {query_type}")