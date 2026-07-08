from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

loader = PyPDFLoader("agentic_ai.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)
chunked_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

vector_store = FAISS.from_documents(chunked_docs, embeddings)

question = "How agentic AI works?"

retrieval = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

retrieved_docs = retrieval.invoke(question)

texts = []

for doc in retrieved_docs:
    texts.append(doc.page_content)

context_text = "\n\n".join(texts)

prompt = PromptTemplate(
    template="you are an help full assistant. give answer only from the provided pdf context. if you are unable to find relevant context from it, just say that you dont know about it. Question - {question}, context - {context}",
    input_variables=['question', 'context']
)

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


final_prompt = prompt.invoke({
    "context" : context_text,
    "question" : question
})

answer = model.invoke(final_prompt)
print(answer.content)