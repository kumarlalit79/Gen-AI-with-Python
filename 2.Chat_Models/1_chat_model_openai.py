from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    max_completion_tokens=10
)

result = model.invoke("Write 5 line poem on ai")

print(result.content)
