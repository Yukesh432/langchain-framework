import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage



os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm= OpenAI()
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# question = "What is deep learning ? How it is different from machine learning?"

# print(llm_chain.run(question))
chat = ChatOpenAI(temperature=0)
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
print(chat(messages))
