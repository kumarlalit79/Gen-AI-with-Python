from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ==========================
# Step 1: Document Ingestion
# ==========================

video_id = "_hmZYKGCkVs"

ytt_api = YouTubeTranscriptApi()
transcript_list = ytt_api.fetch(video_id, languages=["en"])

transcript = " ".join(chunk.text for chunk in transcript_list)

# ==========================
# Step 2: Text Splitting
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

chunks = splitter.create_documents([transcript])

# ==========================
# Step 3: Embeddings + Vector Store
# ==========================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

vector_store = FAISS.from_documents(chunks, embeddings)

# ==========================
# Step 4: Retriever
# ==========================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# ==========================
# Step 5: LLM
# ==========================

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.2,
)

# ==========================
# Step 6: Prompt
# ==========================

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# ==========================
# Step 7: Formatting Function
# ==========================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================
# Step 8: RAG Chain
# ==========================

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

rag_chain = (
    parallel_chain
    | prompt
    | llm
    | parser
)

# ==========================
# Step 9: Ask Question
# ==========================

question = (
    "Why does Gary believe the removal of gatekeepers "
    "has democratized the creator economy?"
)

answer = rag_chain.invoke(question)

print(answer)