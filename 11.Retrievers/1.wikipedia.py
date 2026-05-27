from langchain_community.retrievers import WikipediaRetriever

# Initialize your retriever. 
retriever = WikipediaRetriever(
    top_k_results=2, #apko kitne results chahhiye
    lang="en"
)

# Define your query. 
query = "Tell me about AI agents. "


# Get relevant Wikipedia documents. We are able to use the invoke function. That means retriever is runnable.  
docs = retriever.invoke(query)

print(docs[0].page_content)


# Print Retrieved Content 
for i, doc in enumerate(docs):
    print(f"\n-- Result {i+1} --")
    print(f"Content: \n {doc.page_content} --")