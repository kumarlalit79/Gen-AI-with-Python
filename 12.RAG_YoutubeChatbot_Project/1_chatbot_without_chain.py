from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# step 1 (a) -  Indexing (Document Ingestion)
video_id = "_hmZYKGCkVs"
try:
    ytt_api = YouTubeTranscriptApi()
    
    transcript_list = ytt_api.fetch(video_id, languages=['en'])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    
    # print(transcript)

except Exception as e:
    print("Error:", e)


# step 1 (b) - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)
chunks = splitter.create_documents([transcript])

# checking length of chunks
# print(len(chunks))

# if you want to see some particular chunk. i want to see second chunk
# print(chunks[2])

# step 1 c & d - Indexing(Embedding generation and storing in vector store)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)
    
# if you want to see any particular chunk
# print(vector_store.get_by_ids(['9298effa-950b-4ac2-b2cb-144d83884afb']))


# step 2 - Retrieval
retrieval = vector_store.as_retriever(
    search_type = "similarity", # similarity search karega
    search_kwargs = {"k":2} # and 2 sbse similar vectors nikaal kar ke dega(most relevant doc waala kaam)
)
# print(retrieval)

retrieval_result = retrieval.invoke('What is Gary Vaynerchuks core argument regarding the current state of social media and business opportunity?')
# print(retrieval_result)

# step 3 - augmentation
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.2,
)

prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        
        {context}
        Question: {question}
    """,
    input_variables=['context','question']
)

question = "What is Gary Vaynerchuks core argument regarding the current state of social media and business opportunity?"
retrievad_docs = retrieval.invoke(question)
# print(retrievad_docs)

context_text = "\n\n".join(doc.page_content for doc in retrievad_docs)
# print(context_text)
# ya to upper ki ek line use karlo ya fir neeche ki 4 line
# texts = []
# for doc in retrieved_docs:
#     texts.append(doc.page_content)
# context_text = "\n\n".join(texts)


final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})
# print(final_prompt)


# Step 4 - Generation
answer = llm.invoke(final_prompt)
print(answer.content)