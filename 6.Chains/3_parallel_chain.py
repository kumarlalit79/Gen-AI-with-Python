from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

model1 = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt1 = PromptTemplate(
    template="Generate simple, short, and concise notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate the 5 quiz questions from the following text {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes, and quiz into a single document. \n notes -> {notes}, quiz -> {quiz} ",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()


# firslty create a runnable parallel chain
parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser,
})

# merging logic
merge_chain = prompt3 | model1 | parser

# Now we have to build a final chain by connecting the merge chain and the parallel chain. 
chain = parallel_chain | merge_chain


# Now just we have to invoke the chain. 
text = '''
        Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in engineering, mathematics and computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.[1]

        High-profile applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, and play and analysis in strategy games (e.g., chess and Go). Since the 2020s, generative AI has become widely available to generate images, audio, and videos from text prompts.

        The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics.[a] To reach these goals, AI researchers have used techniques including state space search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[2] Some companies, such as OpenAI, Google DeepMind and Meta, aim to create artificial general intelligence (AGI) – AI that can complete virtually any cognitive task at least as well as a human.[3]
'''


result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()