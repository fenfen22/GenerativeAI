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

# with open('C:\\Users\\difto\\Desktop\\Cours\\F23\\02456 Deep Learning\\Learning objectives.txt', encoding="utf8") as f:
#     documents += f.read()

# with open('C:\\Users\\difto\\Desktop\\Cours\\F23\\02456 Deep Learning\\Course plan.txt', encoding="utf8") as f:
#     documents += f.read()

# # Embedding the documents and storing them in a vector store
# vectorstore = FAISS.from_texts(
#     [documents], embedding=OpenAIEmbeddings()
# )

# # Retriever
# retriever = vectorstore.as_retriever()

# Load, split, embed, and store in vector DB
raw_documents = TextLoader('learning objectives.txt', encoding="utf8").load()
raw_documents += TextLoader('course plan.txt', encoding="utf8").load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# Definition of the system -------------------------------------------------

# PROMPT TEMPLATE
TUTOR_PROMPT = "You are a tutor for a Master's level deep learning course, your role is to understand the course outline and learning objectives thoroughly, as outlined in the provided documentation. Your primary goal is to assist students in grasping the course material, ensuring that their learning experience is both comprehensive and tailored to their individual needs and learning styles. Leverage the existing course material and find innovative ways to adapt these resources to suit the varied learning preferences of your students. This personalized approach is key to enhancing their understanding and application of deep learning concepts."
CONTEXT_PROMPT = "Use the extract from the course material below to answer the user's question:\n{context}"
MEMORY_PROMPT = "Also take into account the past messages from this conversation:\n{history}"

COMPLETE_TEMPLATE = 'system:' + TUTOR_PROMPT + '\n' + CONTEXT_PROMPT + '\n' + MEMORY_PROMPT + '\n' + 'user: {user_input}'

prompt_template = ChatPromptTemplate.from_template(COMPLETE_TEMPLATE)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# CHAT HISTORY
# Short-term memory
stm = []
MEMORY_LENGTH = 5

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
    stm = stm[-(MEMORY_LENGTH-1):] + ["student:"+q+"\n"]#+"\ntutor:"+a+"\n"]

print('Goodbye!')