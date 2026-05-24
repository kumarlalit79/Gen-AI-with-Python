from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 5 pointer Summary about the following text \n. {text}",
    input_variables=['text']
)

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic' : 'how to use AI effectively in 10 lines'})

print(result)