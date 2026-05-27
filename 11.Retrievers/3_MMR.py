from langchain_community.vectorstores import FAISS #Chroma की जगह इस बार यह वाला vector database use कर रहे हैं। 
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

document = [
    Document(page_content="Retrieval augmented generation combines search with language generation."),
    Document(page_content="A vector store saves numerical representations of text called embeddings."),
    Document(page_content="ChromaDB is a lightweight database designed for storing and searching embeddings."),
    Document(page_content="Metadata adds extra information to each document, such as source name, author, or date")
]


embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
)

vectorstore = FAISS.from_documents(
    documents=document,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr", #This enables MMR. 
    search_kwargs={"k":3, "lambda_mult":1} # I want top three results. And lambda mult means relevent diversity balance. 
) 

query = "What is RAG?"
result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"Content: \n {doc.page_content} --")