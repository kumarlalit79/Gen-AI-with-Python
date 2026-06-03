from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# arg schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")
    

# hamari khud ki class - tool
class MultiplyTool(BaseTool):
    name: str = "multiply" #yaha khud ke attribute define kr diye. yaha tool ka naam bata diya
    description: str = "Multiply two numbers"
    
    args_schema: Type[BaseModel] = MultiplyInput #maine bata diya ki schema konsa use karna hai
    
    # naam exactly _run hona chahhiye
    def _run(self, a: int, b: int) -> int:
        return a*b
    
# ab usk tool ka object bana liya
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)