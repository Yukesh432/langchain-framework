import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

api_key= os.environ.get("OPENAI_API_KEY")

# print(api_key)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm= OpenAI(model_name='gpt-4', temperature=0.5, max_tokens=50)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is Narrowband FM?"
print(llm_chain.run(question))