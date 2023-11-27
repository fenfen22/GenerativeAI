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

llm = ChatOpenAI(model="gpt-4", temperature=0)

@tool
def save_student_info(info: str) -> None:
    """
    Saves relevant info about the student (name, student number, courses taken, etc) so it can be retrieved later.
    """
    with open('info.txt', 'w+') as f:
        f.write(info)

@tool
def load_student_info() -> str:
    """
    Loads info about the student (name, student number, courses taken, etc) user to improve the learning experience.
    """
    with open('info.txt', 'r+') as f:
        return f.read()

@tool
def get_learning_objectives() -> str:
    """
    Loads the course's learning objectives as text.
    """
    with open('objectives.txt', 'r+') as f:
        return f.read()

tools = [save_student_info, load_student_info, get_learning_objectives]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a tutor for a Master's level course in deep learning.\
            You have access to the course's learning objectives in your tools.\
            Your goal is to answer questions the student has about the course, and make sure they reach the learning objectives.\
            You can ask questions to assess the student's level in the course.\
            You can and should save info about the student in the text file mentioned in your tools.\
            If no info is available, you can ask questions to learn more about the student.\
            Use that info to tailor your help.",
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print('Hello! I am your personal tutor for the Deep Learning course, ask me something: (type in then press Enter)')

while True:
    print('\nUser:')
    user_input = input()
    if user_input == 'exit': break
    print('\nTutor:')
    print(agent_executor.invoke({"input": user_input}))#['output'])

print('Goodbye!')