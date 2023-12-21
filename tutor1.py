# IMPORTS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

# RAG ----------------------------------------------------------------------

# Loading documents
# documents = ""

# # Embedding the documents and storing them in a vector store
# vectorstore = FAISS.from_texts(
#     [documents], embedding=OpenAIEmbeddings()
# )

# # Retriever
# retriever = vectorstore.as_retriever()

# Load, split, embed, and store in vector DB
raw_documents = TextLoader('owngpt\knowledgeBase\GeneralInformation\LearningObjectives.txt', encoding="utf8").load()
raw_documents += TextLoader('owngpt\knowledgeBase\GeneralInformation\CoursePlan.txt', encoding="utf8").load()
raw_documents += TextLoader('owngpt\knowledgeBase\w1.md', encoding="utf8").load()
raw_documents += TextLoader('owngpt\knowledgeBase\w1.md', encoding="utf8").load()
raw_documents += TextLoader('owngpt\knowledgeBase\w3.4.md', encoding="utf8").load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# Definition of the system -------------------------------------------------

# PROMPT TEMPLATE
TUTOR_PROMPT = "You are a teaching assistant for the course '02456 Deep Learning'. \
    This is a Master's level course taught in English at the Technical University of Denmark. \
    Your goals are to: help students (also referred to as 'user') understand the course material, answer any question they have about it, and ensure that their learning experience is comprehensive and tailored to their individual needs and learning styles. \
    You should adapt your tone so the students enjoy the conversation. \
    Only answer questions related to the course. If the student asks ANY unrelated question, politely refocus on the course. \
    Leverage the existing course material and find innovative ways to adapt these resources to suit the varied learning preferences of the students."
CONTEXT_PROMPT = "Use the extract from the course material below to answer the student's question:\n{context}"
MEMORY_PROMPT = "You should sound conversational, so here are some past messages from your conversation with the student:\n{history}"

COMPLETE_TEMPLATE = 'system:' + TUTOR_PROMPT + '\n' + CONTEXT_PROMPT + '\n' + MEMORY_PROMPT + '\n' + 'user: {user_input}'

prompt_template = ChatPromptTemplate.from_template(COMPLETE_TEMPLATE)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# CHAT HISTORY
# Short-term memory (list of messages)
stm = []
# How many messages to store
MEMORY_LENGTH = 5*2

# FINAL CHAIN
chain = prompt_template | llm | StrOutputParser() # the output parser allows to display only the system's answer

# Interaction with the user ------------------------------------------------
print('Hello! I am your personal tutor for the Deep Learning course, ask me something: (type in then press Enter)')

while True:
    # The user asks a question
    q = input('\nUser: ')
    if q == 'exit': break
    # The question is embedded
    embedding_vector = OpenAIEmbeddings().embed_query(q)
    # The vector store is searched for relevant information
    docs = db.similarity_search_by_vector(embedding_vector)
    # The LLM prompt is augmented using the retrieved information
    context = ""
    for doc in docs:
        print("\nRetrieving information from:" + doc.metadata['source'])
        context += doc.page_content
    # The LLM generates an answer
    chat_history = ''.join(stm)
    a = chain.invoke({"user_input": q, "context": context, "history": chat_history})
    print('\nTutor:' + a)
    # We store the N latest questions from the student to provide short-term memory
    stm = stm[-(MEMORY_LENGTH-1):] + ["student:"+q+"\n"] + ["\ntutor:"+a+"\n"]

print('Goodbye!')