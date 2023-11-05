from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.tools import ShellTool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
# tool= ShellTool()
# print(tool.run("echo this is working"))

llm = OpenAI(temperature=0.5)
tools = load_tools(["wikipedia", "llm-math", "terminal"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
def _should_check(serialized_obj: dict) -> bool:
    # Only require approval on ShellTool.
    return serialized_obj.get("name") == "terminal"


def _approve(_input: str) -> bool:
    if _input == "echo 'Hello World'":
        return True
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")

callbacks = [HumanApprovalCallbackHandler(should_check=_should_check, approve=_approve)]

print(agent.run(
    "Who invented category theory in mathematics?",
    callbacks=callbacks,
))