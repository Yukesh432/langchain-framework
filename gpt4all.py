from langchain.llms import GPT4All

model = GPT4All(model="../LLM models/models/orca-mini-7b.ggmlv3.q4_0.bin", n_threads=8)

# Simplest invocation
response = model("hello. do you know about spiking NN?")

print(response)