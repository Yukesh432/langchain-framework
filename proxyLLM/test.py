from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.agents import initialize_agent


tools= load_tools(['python_repl'])

response= ["Action: Python REPL\nAction Input: print(2+2)", "Final Answer: 4"]
llm= FakeListLLM(responses=response)

agent= initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose= True)

agent.run("Whats 2+2:")