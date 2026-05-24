from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    dimensions=300
)

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli'

# it will return 5 vectors, each in 300D space
doc_embeddings = embeddings.embed_documents(document)

# it will return 1 single vector
query_embeddings = embeddings.embed_query(query)

# calculating similarity
scores = cosine_similarity([query_embeddings], doc_embeddings)[0]
# output was coming in the 2d space, so we wrote 0 so we are getting output in 1d now

index , score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]
# sorting score in descending order taki sbse similar waala jiska score sbse jayada hai wo mil jaye

print(query)
print(document[index])
print("similarity score is : ", score)