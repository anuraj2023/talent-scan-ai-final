import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def ingest(df: pd.DataFrame, content_column: str, embedding_model, batch_size: int = 1000):
    try:
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")

        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Sample data from '{content_column}': {df[content_column].head()}")

        loader = DataFrameLoader(df, page_content_column=content_column)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len
        )
        documents = loader.load()
        logging.info(f"Number of documents loaded: {len(documents)}")

        if not documents:
            raise ValueError("No documents were loaded. Please check your DataFrame and content column.")

        document_chunks = text_splitter.split_documents(documents)
        logging.info(f"Number of document chunks: {len(document_chunks)}")

        if not document_chunks:
            raise ValueError("No document chunks were created. Please check your text splitter settings.")

        # Initialize FAISS with the first batch to avoid empty list error
        first_batch = document_chunks[:batch_size]
        vectorstore_db = FAISS.from_documents(first_batch, embedding_model, distance_strategy=DistanceStrategy.COSINE)

        # Add remaining documents in batches
        for i in tqdm(range(batch_size, len(document_chunks), batch_size)):
            batch = document_chunks[i:i+batch_size]
            try:
                vectorstore_db.add_documents(batch)
            except Exception as e:
                logging.error(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                logging.error(f"Problematic batch: {batch}")
                raise

        logging.info("Ingestion completed successfully")
        return vectorstore_db

    except Exception as e:
        logging.error(f"Error during ingestion: {str(e)}")
        raise
