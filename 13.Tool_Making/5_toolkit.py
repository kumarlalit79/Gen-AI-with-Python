from langchain_core.tools import tool

# 2 custom tools
@tool
def add(a: int, b: int) -> int : 
    """Add two numbers"""
    return a+b

@tool
def multiply(a: int, b: int) -> int : 
    """Multiply two numbers"""
    return a*b


# making toolkit -> class banake karte hai
class MathToolKit:
    def get_tools(self):
        return [add, multiply]
    

# now just make obj of your toolkit
toolkit = MathToolKit()

tools = toolkit.get_tools()

# loop chala ke in tools ko access kar sakte hai
for tool in tools:
    print(tool.name, "=>", tool.description)