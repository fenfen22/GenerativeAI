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