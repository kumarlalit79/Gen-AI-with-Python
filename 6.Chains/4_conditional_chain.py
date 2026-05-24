from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

model1 = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

# pydantic response 
class FeedbacK(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback.')
    
parser2 = PydanticOutputParser(pydantic_object=FeedbacK)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the feedback into : positive or negative. \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

# Firstly, let's develop the classifier chain which will classify the feedback into positive or negative. 
classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template='Write appropriate response to this positive feedback \n {feedback} ',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write appropriate response to this negative feedback \n {feedback} ',
    input_variables=['feedback']
)

# Now creating branch in 
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser), #this is if condn
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x:'Could not find sentiment. ')
)


# Now I have my branch chain and my classifier chain. Now I will merge them both and create a final chain. 
chain = classifier_chain | branch_chain
result = chain.invoke({'feedback' : 'this is a very good laptop'})
print(result)