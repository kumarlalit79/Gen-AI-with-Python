from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the joke {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'AI'})
print(result)