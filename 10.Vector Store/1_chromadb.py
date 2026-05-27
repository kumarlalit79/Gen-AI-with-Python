from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

# creating document objects 
doc1 = Document(
    page_content="LangChain helps developers build applications that connect language models with external data sources, tools, and workflows. It provides reusable components for prompts, chains, memory, retrievers, and agents, making it easier to create practical AI systems instead of isolated model calls.",
    metadata={"source": "langchain_overview", "topic": "langchain"}
)

doc2 = Document(
    page_content="A vector store saves numerical representations of text called embeddings. These embeddings allow an application to search by meaning rather than exact keywords. When a user asks a question, the query is embedded and compared with stored vectors to find the most relevant documents.",
    metadata={"source": "vector_store_notes", "topic": "embeddings"}
)

doc3 = Document(
    page_content="ChromaDB is a lightweight database designed for storing and searching embeddings. It is commonly used in retrieval augmented generation applications where documents are embedded, stored in a collection, and later retrieved as context for a language model.",
    metadata={"source": "chromadb_intro", "topic": "chromadb"}
)

doc4 = Document(
    page_content="Text splitting is important because language models and embedding models work best with manageable chunks of content. A good splitter keeps related ideas together while avoiding chunks that are too large, which improves the quality of retrieval results.",
    metadata={"source": "text_splitting_guide", "topic": "text_splitting"}
)

doc5 = Document(
    page_content="Metadata adds extra information to each document, such as source name, category, author, or date. This information can be used to filter search results, organize documents, and make retrieval more accurate for specific user queries.",
    metadata={"source": "metadata_notes", "topic": "metadata"}
)

doc6 = Document(
    page_content="Retrieval augmented generation combines search with language generation. Instead of relying only on the model's training data, the application retrieves relevant documents from a vector store and passes them to the model as fresh context.",
    metadata={"source": "rag_summary", "topic": "rag"}
)


docs = [doc1, doc2, doc3, doc4, doc5]

# creating vector db
vector_store = Chroma(
    
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    
    ), #सबसे पहले बताना होता है कि जब document object को हम अपने database के अंदर insert करेंगे, तो कौन सा model हम use करेंगे embedding generate करने के लिए? 
    
    persist_directory='chroma_db',#आप जो documents vector की form में store करना चाहते हो, वो किस location में store होगा? तो हम Chroma DB नाम का एक folder बनाएंगे। उसके अंदर सारा data store करेंगे। 
    
    collection_name='sample' #As we know, Chroma DB के अंदर हम एक collection बनाते हैं, तो उस collection का नाम हम sample रख रहे 
)


# add documents
vector_store.add_documents(docs)

print("Documents Added Successfully\n")


# view documents
data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print(data)

# search documents
vector_store.similarity_search(
    query='Who amongs these are database?',
    k=2
)

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who amongs these are database?',
    k=2
)

# meta data filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"source": "text_splitting_guide"}
)

# update existing documents
updated_doc1 = Document(
    page_content="write new data here",
    metadata={"source": "text_splitting_guide"}
)

vector_store.update_document(document_id="write id here" , document=updated_doc1)


# delete particular document
vector_store.delete(ids=['write id here'])