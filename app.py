import os
import traceback
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS

# Keep your original import path & names
from QASystem.ingestion import data_ingestion, get_vector_store
from QASystem.retrievalandgeneration import get_claude_llm, get_response_llm

# Use session so it respects AWS_PROFILE/AWS_REGION from the same shell
_session = boto3.Session()
_bedrock = _session.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "eu-west-2"))
_bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=_bedrock)

def main():
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using LangChain and AWS Bedrock (Claude 3.7 Sonnet)")

    user_question = st.text_input("Ask a question from the PDF files")

    with st.sidebar:
        st.title("Update or create the vector store")

        if st.button("Vectors Update"):
            try:
                with st.spinner("Processing PDFs..."):
                    docs = data_ingestion()
                    get_vector_store(docs)
                st.success("Vector store rebuilt ✔")
            except Exception as e:
                st.error("Failed to rebuild vectors. Check AWS creds/region and data folder.")
                st.code("".join(traceback.format_exc()))
                st.stop()

        if st.button("Claude Model"):
            if not user_question.strip():
                st.warning("Please enter a question first.")
            else:
                try:
                    with st.spinner("Thinking..."):
                        faiss_index = FAISS.load_local(
                            "faiss_index",
                            _bedrock_embeddings,
                            allow_dangerous_deserialization=True,
                        )
                        llm = get_claude_llm()
                        st.write(get_response_llm(llm, faiss_index, user_question))
                        st.success("Done ✔")
                except Exception as e:
                    st.error("Query failed. Verify Bedrock model availability/permissions and the FAISS index.")
                    st.code("".join(traceback.format_exc()))
                    st.stop()

if __name__ == "__main__":
    main()
