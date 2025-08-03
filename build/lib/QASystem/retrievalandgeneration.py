from langchain.chains import retrieval_qa
from langchain.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
import boto3 
from langchain.prompts import PromptTemplate
from QASystem.ingestion import get_vector_store

bedrock=boto3.client(service_name="bedrock-runtime")



prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)


def get_claude3_llm():
    llm=Bedrock(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",client=bedrock, model_kwargs={"maxTokens":512})
    return llm






def get_response_llm(llm,vectorstore_faiss,query):
    qa=retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}

    )
    answer=qa({"query":query})
    return answer["result"]

if __name__=='__main__':
    vectorstore_faiss=get_vector_store()
    query="What is RAG token?"
    llm=get_claude3_llm()
    get_response_llm(llm,vectorstore_faiss,query)