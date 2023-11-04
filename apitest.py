import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

OPENAI_API_KEY= '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm= OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

print(llm_chain.run(question))