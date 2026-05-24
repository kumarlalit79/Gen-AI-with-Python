from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_classic.schema.output_parser import StrOutputParser
import os

load_dotenv()

prompt = PromptTemplate(
    template="Summaries the text in 5 small points \n {text}",
    input_variables=['text']
)

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

parser = StrOutputParser()

loader = TextLoader('ai_wikkipedia.txt', encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'text' : docs[0].page_content})
print(result)