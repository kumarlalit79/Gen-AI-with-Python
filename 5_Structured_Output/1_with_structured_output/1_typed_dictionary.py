from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import TypedDict

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# schema
class Review(TypedDict):
    summary: str
    sentiment: str
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(type(result))
print(result)