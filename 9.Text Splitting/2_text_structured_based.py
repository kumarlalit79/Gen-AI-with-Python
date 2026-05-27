from langchain_text_splitters import RecursiveCharacterTextSplitter

text = '''

    AI agents possess several key attributes, including goal-directed behavior, natural language interfaces, the capacity to use external tools, and the ability to perform multi-step tasks. Their control flow is frequently driven by large language models (LLMs). Agent systems may also include memory components, planning logic, tool interfaces, and orchestration software for coordinating agent components.[2][3]

    AI agents do not have a standard definition.[4][5][6] NIST describes agentic AI as an emerging area requiring standards for secure operation, interoperability, and reliable interaction with external systems.[1]

    A common application of AI agents is task automation: for example, booking travel plans based on a user's prompted request.[7][8][9]

    Companies such as Google, Microsoft and Amazon Web Services have offered platforms for deploying pre-built AI agents.[10] Several protocols have been proposed for standardizing inter-agent communication, with examples including the Model Context Protocol, Gibberlink,[11] and many others. Some of these protocols are also used for connecting agents to external applications.[12]

    In December 2025, Linux Foundation announced the formation of the Agentic AI Foundation (AAIF), with the goal of ensuring agentic AI evolves transparently and collaboratively.[13][14]

'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)