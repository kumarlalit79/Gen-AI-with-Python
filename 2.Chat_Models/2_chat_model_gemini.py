from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

result = model.invoke("give me 5 best places of india (no other fluf, just 5 place)")

print(result.content)