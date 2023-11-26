from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import ConversationChain
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor

'''

Following and adapting the example from: https://python.langchain.com/docs/modules/agents/

Goal is to make a basic tutor who can store and retrieve information about a student.

'''

# Remove this before pushing code changes to git
OPENAI_API_KEY = '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'

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

@tool
def save_student_info(info: str) -> None:
    """
    Saves information about the student user in a text file so it can be retrieved later.
    Information like name, student number, courses taken, etc.
    """
    with open('info.txt', 'a') as f:
        f.write(info)

@tool
def load_student_info() -> str:
    """
    Loads information about the student user from a text file to improve the conversation.
    Information like name, student number, courses taken, etc.
    """
    with open('info.txt', 'r') as f:
        return f.read()

tools = [save_student_info, load_student_info]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a university tutor, your goal is to help the student user with their learning. You can and should save info about the student in the text file in your tools.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

print('I am your personal tutor, ask me something: ')

while True:
    user_input = input()
    if user_input == 'exit': break
    agent_executor.invoke({"input": user_input})

print('Goodbye!')



