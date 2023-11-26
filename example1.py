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

Following the example from: https://python.langchain.com/docs/modules/agents/

'''

OPENAI_API_KEY = ''

#chat = ChatOpenAI(api_key=OPENAI_API_KEY)

#conversation = ConversationChain(llm=chat)

#conversation.run("Translate this sentence from English to French: I love programming.")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
#r = llm.invoke("how many letters in the word educa?")
#print(r)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a very powerful assistant, but bad at calculating lengths of words.",
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

#r = agent.invoke({"input": "how many letters in the word educa?", "intermediate_steps": []})
#print(r)

""" user_input = "how many letters in the word educa?"
intermediate_steps = []
while True:
    output = agent.invoke(
        {
            "input": user_input,
            "intermediate_steps": intermediate_steps,
        }
    )
    if isinstance(output, AgentFinish):
        final_result = output.return_values["output"]
        break
    else:
        print(f"TOOL NAME: {output.tool}")
        print(f"TOOL INPUT: {output.tool_input}")
        tool = {"get_word_length": get_word_length}[output.tool]
        observation = tool.run(output.tool_input)
        intermediate_steps.append((output, observation))
print(final_result) """

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "how many letters in the word educa?"})





