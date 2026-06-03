from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# pydantic model
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

# function
def multiply_function(a: int, b: int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply_function, #hamara fun kya hai jiski help se hame tool banana hai
    name="multiply", #us function ko ham ek naam denge
    description="multiply two numbers", #ek desc bhi
    args_schema=MultiplyInput #pydantic class
)

result = multiply_tool.invoke({"a": 3, "b": 3})
print(result)