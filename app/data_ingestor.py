import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader

def ingest(df: pd.DataFrame, content_column: str, embedding_model):
  loader = DataFrameLoader(df, page_content_column=content_column)

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=300  # Reduced overlap to minimize duplicates
  )
  documents = loader.load()
  document_chunks = text_splitter.split_documents(documents)

  vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.DOT_PRODUCT)
  return vectorstore_db