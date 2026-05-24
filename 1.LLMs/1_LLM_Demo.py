from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# creating obj of OpenAI and telling it that we want to communicate with gpt-4o-mini model
llm = OpenAI(
    model='gpt-4o-mini',
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = "https://models.inference.ai.azure.com"
)


# invoke is langchain method, it takes prompt. with the help of this, we can communicate with llm models
result = llm.invoke("what is capital of india")

print(result)