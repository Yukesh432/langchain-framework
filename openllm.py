# from langchain.llms import OpenLLM
# llm = OpenLLM(
#     model_name='flan-t5',
#     model_id='google/flan-t5-large',
# )
# output= llm("What is the difference between a duck and a goose?")
# print(output)

import openllm
client = openllm.client.HTTPClient('http://localhost:3000')
client.query('Explain to me the difference between "further" and "farther"')