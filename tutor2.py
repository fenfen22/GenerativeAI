from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor, initialize_agent, tool, Tool
from langchain.vectorstores import Chroma
import streamlit as st
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
import json
from pathlib import Path
from langchain.vectorstores.redis import Redis
from langchain.memory import RedisChatMessageHistory

'''

Following and adapting the example from: https://python.langchain.com/docs/modules/agents/

Goal is to make a basic tutor who can store and retrieve information about a student.

'''

token = 'AZ5QACQgMjY0MDY0MDQtM2YwMC00ODFhLTlhNDEtNWU1MDBjNDYxZTNmMWJhMjk2YjZmNmNjNDRlY2E1OTM3NTY3MmM5YWZjYTI='

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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
    
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

knowledgeBase = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",retriever=db.as_retriever()
)

knowledgeBaseTool = Tool(
        name="KnowledgeBase",
        func=knowledgeBase.run,
        description="Contains all the Deep Learning course content.",
    )

tools = [save_student_info, load_student_info, get_learning_objectives, knowledgeBaseTool]

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

# https://python.langchain.com/docs/modules/memory/agent_with_memory_in_db  link of redisChatMessageHistory

# ttl:600 seconds later, this session will get expired
message_history = RedisChatMessageHistory(url='redis://default:1ba296b6f6cc44eca59375672c9afca2@us1-brave-lynx-40528.upstash.io:40528',ttl=600, session_id="my-session")
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)
# agent_chain = initialize_agent(tools=tools, llm=llm, agent=agent,memory=memory,verbose=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=memory)


# app framework
st.header("🦜️🔗 Personal Learning Assitants")

user_input = st.text_input("You: ")

if "memory" not in st.session_state:
    st.session_state["memory"] = []

if st.button("Submit"):
    if user_input:
        with st.spinner("Generating response..."):
            response = agent_executor({"input":user_input}, return_only_outputs=True)
            st.write(response['output'])
            st.session_state["memory"].append(memory.chat_memory.messages)
    else:
        st.warning("Please enter your question")

chat_history = []
for message in st.session_state["memory"]:
    ingest_to_db = messages_to_dict(message)
    chat_history.append(ingest_to_db)



with Path("message.json").open("w") as f:
    json.dump(chat_history,f)

# # code for running this: streamlit run tutor2.py