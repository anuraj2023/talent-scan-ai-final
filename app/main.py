import sys, os
from s3_ingestor import upload_to_s3
import io 
import pandas as pd
import streamlit as st
import openai
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from logging_config import logger
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import time
from llm_agent import ChatBot
from  data_ingestor import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity
import os
from config import get_env_vars

sys.dont_write_bytecode = True
env_vars = get_env_vars()
OPEN_AI_KEY = env_vars['OPEN_AI_KEY']
S3_BUCKET_NAME = env_vars['S3_BUCKET_NAME']

def get_openai_model_names():
    url = "https://api.openai.com/v1/models"
    api_key = OPEN_AI_KEY
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        models = response.json()["data"]
        gpt_model_names = [model["id"] for model in models if "gpt" in model["id"].lower()]
        print("model_names are: ", gpt_model_names)
        return gpt_model_names
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# List of available Open AI models
AVAILABLE_MODELS = [
   "gpt-4o-mini-2024-07-18",
   "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct"
]


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
  2. Type in a job description query. 💬 (Example: Give me candidates who has worked on LLM)
"""

# Set up Streamlit page
st.set_page_config(page_title="Talent Scan AI")
st.title("Talent Scan AI")

# Initialize session state variables
if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS[0]

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

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = DocxDocument(docx_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def process_uploaded_files(uploaded_files, file_type):
    """Process uploaded resume or job description files and update the vectorstore."""
    s3_folder = "resumes" if file_type == "resume" else "job-descriptions"
    
    if uploaded_files:
        df_list = []
        new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]
        
        if not new_files:
            return 

        for uploaded_file in new_files:
            try:
                if uploaded_file.type == "application/pdf":
                    file_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    file_text = extract_text_from_docx(io.BytesIO(uploaded_file.getvalue()))
                else:
                    st.error(f"Uploaded file {uploaded_file.name} is not a PDF or DOCX.")
                    logger.warning(f"Skipped unsupported file: {uploaded_file.name}")
                    continue

                df_list.append(pd.DataFrame({"ID": [uploaded_file.name], "Content": [file_text]}))
                
                # Upload to S3
                s3_file_name = f"{s3_folder}/{uploaded_file.name}"
                status, url = upload_to_s3(io.BytesIO(uploaded_file.getvalue()), S3_BUCKET_NAME, s3_file_name)
                if status:
                    logger.info(f"Uploaded {uploaded_file.name} to S3 bucket {S3_BUCKET_NAME}")
                    st.success(f"Uploaded {uploaded_file.name} to S3")
                    st.session_state.processed_files.add(uploaded_file.name)
                else:
                    logger.error(f"Failed to upload {uploaded_file.name} to S3")
            except Exception as error:
                st.error(f"Error processing file {uploaded_file.name}: {str(error)}")
                logger.error(f"Error processing file {uploaded_file.name}: {str(error)}")
        
        # Upload resumes only to vector DB
        if file_type == "resume":
          if df_list:
              with st.spinner(f'Indexing the uploaded {file_type} data. This may take a while...'):
                  df = pd.concat(df_list, ignore_index=True)
                  st.session_state.vectordb = ingest(df, "Content", st.session_state.embedding_model)
                
                  st.session_state.rag_pipeline = SelfQueryRetriever(
                    st.session_state.vectordb, 
                    df
                  )
                  st.session_state.retriever = st.session_state.rag_pipeline
                  st.session_state.llm = ChatBot(
                    api_key=st.session_state.api_key,
                    model=st.session_state.gpt_selection,
                  )
              st.success(f"Successfully processed {len(df_list)} new {file_type} files.")
              logger.info(f"Indexed {len(df_list)} new {file_type} files")

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

def update_selected_model():
    st.session_state.selected_model = st.session_state.gpt_selection
    logger.info(f"Updated selected model to: {st.session_state.selected_model}")


# Set up the main chat interface
user_query = st.chat_input("Type your message here...")

# Set up the sidebar
with st.sidebar:
    st.markdown("# Settings")

    st.text_input("OpenAI's API Key", type="password", key="api_key", value=OPEN_AI_KEY)
    st.selectbox("GPT Model", AVAILABLE_MODELS, key="gpt_selection", on_change=update_selected_model)
    
    uploaded_resumes = st.file_uploader("Upload resumes", type=["pdf"], accept_multiple_files=True)
    if uploaded_resumes:
        process_uploaded_files(uploaded_resumes, "resume")
    
    uploaded_job_description = st.file_uploader("Upload job description", type=["pdf", "docx"], accept_multiple_files=False)
    if uploaded_job_description:
        process_uploaded_files([uploaded_job_description], "job_description")
    
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

if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
      start = time.time()
      with st.spinner("Generating answers..."):
        document_list = st.session_state.retriever.retrieve_docs(user_query, st.session_state.llm, st.session_state.rag_selection)
        query_type = st.session_state.retriever.meta_data["query_type"]
        # st.session_state.resume_list = document_list
        stream_message = st.session_state.llm.generate_message_stream(user_query, document_list, st.session_state.chat_history, query_type)
      end = time.time()

      response = st.write_stream(stream_message)
    
      retriever_message = chatbot_verbosity
      retriever_message.render(document_list, st.session_state.retriever.meta_data, end-start)

      st.session_state.chat_history.append(AIMessage(content=response))
      st.session_state.chat_history.append((retriever_message, document_list, st.session_state.retriever.meta_data, end-start))

      logger.info(f"Processed query in {end-start:.2f} seconds. Query type: {query_type}")