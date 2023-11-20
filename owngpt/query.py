from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle



# def load_retriever():
#     with open("vectorstore.pkl", "rb") as f:
#         vectorstore = pickle.load(f)
#     retriever = VectorStoreRetriever(vectorstore=vectorstore)
#     return retriever

# a=load_retriever()
# print(a)

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# query = "What is the main topic of this document?"
query ="What is PyTorch?"
# query = "What is CIFAR-10 dataset?"
docs = vectorstore.similarity_search(query)
print("length of document:", len(docs))
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
# print(docs[0].page_content)
# print(docs[0].metadata)
# print(docs)

