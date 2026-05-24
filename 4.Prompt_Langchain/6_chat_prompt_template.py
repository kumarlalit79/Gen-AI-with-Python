from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system' , 'You are an helpful {domain} expert'),
    ('human' , 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket', 'topic': 'Carrom ball'})
print(prompt)

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

