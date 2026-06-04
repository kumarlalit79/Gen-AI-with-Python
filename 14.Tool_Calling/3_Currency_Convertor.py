from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
import os
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json

load_dotenv()

# Step 1 - Create Tool
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency 
    """
    url = f'https://v6.exchangerate-api.com/v6/98afc5a0bc89a65dfa59f91c/pair/{base_currency}/{target_currency}'
    
    # we will use req module to send http req to this api
    response = requests.get(url)

    return response.json()

# print(get_conversion_factor.invoke({'base_currency': 'USD', 'target_currency':'INR'}))

@tool
def convert(base_currecy_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """ 
    given a currency conversion rate this function calculates the target currency value from  a given
    """
    return base_currecy_value * conversion_rate

# print(convert.invoke({'base_currecy_value':10, 'conversion_rate': 90}))


# Step 2 - Tool Binding
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between USD and INR, and based on that can you convert 10 usd to inr')]
# print(messages)

ai_message = llm_with_tools.invoke(messages)
# print(ai_message)
# print(ai_message.tool_calls)
messages.append(ai_message)


# executing tool call
for tool_call in ai_message.tool_calls:
    
    # execute the 1st tool and get the value of conversion rate
    if tool_call["name"] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        
        # fetch the converison rate from tool_messaeg1
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        
        # append this tool message to messages list
        messages.append(tool_message1)
    
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call["name"] == 'convert':
        
        # fetch the currenct arg
        tool_call['args']['conversion_rate'] = conversion_rate   
        
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)
        

final_result = llm_with_tools.invoke(messages).content
print(final_result)