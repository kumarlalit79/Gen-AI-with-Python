from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

chat_history = []

# infinite chalega ye chatbot
while True:
    user_input = input('👤 : ')
    chat_history.append(user_input)
    
    if user_input == "exit":
        break
    
    result = model.invoke(chat_history)
    
    chat_history.append(result.content)
    
    print("🤖 : " , result.content)

print(chat_history)