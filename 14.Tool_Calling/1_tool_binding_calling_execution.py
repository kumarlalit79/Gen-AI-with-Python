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
# print(multiply.name)
# print(multiply.description)
# print(multiply.args) #means tool kis input format mai cheeze accept kar raha hai


# tool binding
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

llm_with_tools = llm.bind_tools([multiply])

# tool calling
# result = llm_with_tools.invoke('hi how are you')
# result = llm_with_tools.invoke('Can you multiply 3 with 10')
result = llm_with_tools.invoke('Can you multiply 3 with 10')


# tool execution
tool_exe = multiply.invoke(result.tool_calls[0]['args'])
print(tool_exe)

# sending full result
print(multiply.invoke(result.tool_calls[0]))
