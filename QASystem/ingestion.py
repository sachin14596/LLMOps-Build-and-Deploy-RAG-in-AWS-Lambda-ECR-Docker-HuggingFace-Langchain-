import os
import boto3
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings

# Use the active shell profile/region; default to eu-west-2 if env not set.
_session = boto3.Session()
_bedrock = _session.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "eu-west-2"))

# Amazon Titan Text Embeddings V2
_bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=_bedrock,
)

def data_ingestion():
    """
    Loads PDFs from ./data and splits into chunks.
    """
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    # Use ~20% overlap to avoid duplicates/loops
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    """
    Build and persist FAISS index locally under ./faiss_index
    """
    vector_store_faiss = FAISS.from_documents(docs, _bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss

# Optional: expose these for other modules
bedrock = _bedrock
bedrock_embeddings = _bedrock_embeddings

if __name__ == "__main__":
    _docs = data_ingestion()
    get_vector_store(_docs)
