import os
import sys
import json
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS

from QASystem.ingestion import data_ingestion, get_vector_store
from QASystem.retrievalandgeneration import get_claude_llm, get_response_llm

# Bedrock client (EU North 1)
bedrock = boto3.client(service_name="bedrock-runtime", region_name="eu-west-2")

# Amazon Titan Text Embeddings V2
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock,
    # model_kwargs={"dimensions": 1024, "normalize": True},  # optional
)

def main():
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using LangChain and AWS Bedrock (Claude 3.7 Sonnet)")

    user_question = st.text_input("Ask a question from the PDF files")

    with st.sidebar:
        st.title("Update or create the vector store")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store rebuilt")

        if st.button("Claude Model"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_claude_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

if __name__ == "__main__":
    main()
