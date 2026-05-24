from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# step 1 - jo input bhejna hai LLM ko wo likh liya list mai
messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about langchain')
]

# step 2 - fir usko model ke pass bhej diya, model ne hamko ek result diya
result = model.invoke(messages)

# step 3 - us result ko hamne ek ai message mai convert kr diya
ai_msg = AIMessage(content=result.content)

# step 4 - chat history maintain
messages.append(ai_msg)

# step 5 - chat history print
print(messages)