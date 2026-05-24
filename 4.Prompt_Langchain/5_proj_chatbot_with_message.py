from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('👤 : ')
    
    # convert our user input to human message - human msg ka label laga diya ispe
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == "exit":
        break
    
    result = model.invoke(chat_history)
    
    # converting result to ai msg - ai msg ka label
    chat_history.append(AIMessage(result.content))
    
    print("🤖 : " , result.content)

print(chat_history)