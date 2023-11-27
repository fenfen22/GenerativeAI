from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent, AgentExecutor, AgentOutputParser, LLMSingleActionAgent,Tool
from langchain.llms import OpenAI, OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import re
from typing import Union
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.utilities import SerpAPIWrapper


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

knowledgeBase = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",retriever=db.as_retriever()
)

# set up tools
tools = [
    Tool(
        name="KnowledgeBase QA System",
        func=knowledgeBase.run,
        description="Useful for when you need to answer questions about the specific course content.",
    )
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "Can you provide some suggestions to improve the model’s performance based on the notebook 4.2?"
)

