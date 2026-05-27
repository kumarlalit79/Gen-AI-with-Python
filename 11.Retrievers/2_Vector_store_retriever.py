from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

# Step one: Your source document 
document = [
    Document(page_content="Retrieval augmented generation combines search with language generation."),
    Document(page_content="A vector store saves numerical representations of text called embeddings."),
    Document(page_content="ChromaDB is a lightweight database designed for storing and searching embeddings."),
    Document(page_content="Metadata adds extra information to each document, such as source name, author, or date")
]

# Steo two: Initialize Embedding Model 
embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
)

# Step 3: Create Chroma Vector Store in memory
vectorstore = Chroma.from_documents(
    documents=document,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Step 4: Convert Vector Store into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":2}) #हम को कितने results चाहिए K:2 means?

query = "What is chroma used for?" 
result = retriever.invoke(query)


for i, doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"Content: \n {doc.page_content} --")