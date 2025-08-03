from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock          # <<< switch to ChatBedrock (not Converse)
from langchain_community.embeddings import BedrockEmbeddings
import boto3

from QASystem.ingestion import get_vector_store, data_ingestion

# Bedrock client (EU North 1) — for embeddings
bedrock = boto3.client(service_name="bedrock-runtime", region_name="eu-north-1")

# Amazon Titan Text Embeddings V2
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock,
    # model_kwargs={"dimensions": 1024, "normalize": True},  # optional
)

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_claude_llm():
    # NOTE: Do NOT pass a boto3 client to avoid circular serialization.
    # ChatBedrock uses Bedrock Messages API (sends `messages`, not `prompt`).
    return ChatBedrock(
        model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name="eu-north-1",
        model_kwargs={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.999,
        },
    )

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer["result"]

if __name__ == '__main__':
    # docs = data_ingestion()
    # vectorstore_faiss = get_vector_store(docs)
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    query = "What is RAG token?"
    llm = get_claude_llm()
    print(get_response_llm(llm, faiss_index, query))
