tutor:
explain_concpts
create practical problems 




1. do we have other roles? like student 
2. what kinds of actions do we have for different roles?
3. where is our own data information?
4. OPENAI_API_KEY not avaliable
5. acessing , what level , for grading or giving  credits

6. reasoing the dialog
selfreflect
credits
personalizer learning 
make visulazation
goal: 
step back prompting
not beyong this msterial

reasoning the diale; simulation engines

valuated ? 
fine tuning

langsmis

constrains on prompting template ()

digital twins memory 



stuck at vectorstores and embedding, retrieval

finetune



# import langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# bring in streamlit for UI dev
import streamlit as st
# bring in watsonx interface
from watsonxlangchain import LangChainInterface