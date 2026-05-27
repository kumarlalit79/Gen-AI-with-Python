from langchain_community.document_loaders import WebBaseLoader

url = "https://www.flipkart.com/secrets-millionaire-mind-mastering-inner-game-wealth/p/itmezunpudayeanz?pid=9783845723617&lid=LSTBOK9783845723617V3Z9GK&hl_lid=&marketplace=FLIPKART&fm=eyJ3dHAiOiJyZWNvIiwicHJwdCI6InBwIiwibWlkIjoicHJvZHVjdFJlY29tbWVuZGF0aW9uL3NpbWlsYXIifQ%3D%3D&pageUID=1779647004615"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs)