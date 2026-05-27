from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = '''

    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import os

    load_dotenv()

    prompt = PromptTemplate(
        template="Generate 5 interesting fact about {topic}",
        input_variables=['topic']
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # ye output ko string mai lake dega
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({'topic':'cricket'})

    print(result)

    # visualize graph
    # chain.get_graph().print_ascii()

'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)  