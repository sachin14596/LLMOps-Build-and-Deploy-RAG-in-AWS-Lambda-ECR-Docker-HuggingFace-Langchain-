import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock  # Messages API
from langchain_community.embeddings import BedrockEmbeddings

# Keep your original import path (object used only for FAISS load smoke test)
from QASystem.ingestion import bedrock_embeddings as _bedrock_embeddings

# Prompt (tidied but same intent)
_prompt_template = """
Use the following context to answer the user's question accurately.
If the answer is not in the context, say you don't know.
Provide a clear, structured summary (~200–300 words).

<context>
{context}
</context>

Question: {question}
"""

PROMPT = PromptTemplate(
    template=_prompt_template,
    input_variables=["context", "question"],
)

def get_claude_llm():
    """
    Returns a ChatBedrock LLM for Claude 3.7 Sonnet in eu-west-2.
    - If BEDROCK_CLAUDE37_PROFILE_ARN is set, invoke via that *inference profile ARN*.
      (Pass provider='anthropic' as required by langchain_aws when using an ARN.)
    - Otherwise, invoke via direct model id.
    """
    region = os.getenv("AWS_REGION", "eu-west-2")
    profile_arn = os.getenv("BEDROCK_CLAUDE37_PROFILE_ARN", "").strip()

    model_kwargs = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.999,
    }

    if profile_arn:
        # IMPORTANT: provider is required when model_id is an ARN
        return ChatBedrock(
            model_id=profile_arn,
            provider="anthropic",
            region_name=region,
            model_kwargs=model_kwargs,
        )

    # Direct model id path (works if your account supports direct invocation in this region)
    return ChatBedrock(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name=region,
        model_kwargs=model_kwargs,
    )

def get_response_llm(llm, vectorstore_faiss, query: str) -> str:
    """
    Build a RetrievalQA chain and return only the final answer text.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        ),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT},
    )
    result = qa({"query": query})
    return result["result"]

if __name__ == "__main__":
    # Local smoke test (requires existing ./faiss_index)
    faiss_index = FAISS.load_local(
        "faiss_index",
        _bedrock_embeddings,
        allow_dangerous_deserialization=True,
    )
    llm = get_claude_llm()
    print(get_response_llm(llm, faiss_index, "What is RAG token?"))
