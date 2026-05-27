from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    DirectoryLoader
)

# PDF Loader
pdf_loader = DirectoryLoader(
    path="directories_loader_files", #यह बताता है कि कौन से folders में file present हैं। 
    glob="*.pdf", #यह बताता है कि कौन सी files को load करना है। 
    loader_cls=PyPDFLoader
)

# TXT Loader
txt_loader = DirectoryLoader(
    path="directories_loader_files",
    glob="*.txt",
    loader_cls=TextLoader
)

# CSV Loader
csv_loader = DirectoryLoader(
    path="directories_loader_files",
    glob="*.csv",
    loader_cls=CSVLoader
)

# Load all docs
docs = []

# extend() is a Python list method used to add multiple items from another list into an existing list.
docs.extend(pdf_loader.load())
docs.extend(txt_loader.load())
docs.extend(csv_loader.load())

print(len(docs))
