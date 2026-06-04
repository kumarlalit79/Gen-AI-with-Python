from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


# creating tool
@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their products"""
    return a*b

# print(multiply.invoke({"a": 3, "b": 4}))

# tool binding
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

llm_with_tools = llm.bind_tools([multiply])

query = HumanMessage('Can you multiply 3 with 10')
message = [query]

result = llm_with_tools.invoke(message)
message.append(result)

tool_result = multiply.invoke(result.tool_calls[0])
message.append(tool_result)

# print(llm_with_tools.invoke(message))
print(llm_with_tools.invoke(message).content)