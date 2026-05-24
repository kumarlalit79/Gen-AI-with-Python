from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    max_completion_tokens=10
)

# heading
st.header("Research Tool")

# creating input box
user_input = st.text_input('Enter your prompt')

# creating button
if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content) 