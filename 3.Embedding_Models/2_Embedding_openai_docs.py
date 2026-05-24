from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    dimensions=32
)

documents = [
    "Delhi is the capital of India",
    "I live in Uttarakhand",
    "I am learning python and generative ai"
]

result = embeddings.embed_documents(documents)

print(str(result))