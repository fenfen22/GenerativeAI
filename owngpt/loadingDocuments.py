

import getpass
import openai
from pathlib import Path
import os
import pandas as pd
from langchain.document_loaders import (
    TextLoader,
    NotebookLoader,
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import faiss

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = input("OpenAI API Key:")


def read_data(data_path: Path):
    suffix = data_path.suffix
    if '.xlsx' == suffix:
        data = pd.read_excel(data_path)
    elif '.csv' == suffix:
        data = pd.read_csv(data_path)
    elif '.json' == suffix:
        data = pd.read_json(data_path)
    elif suffix in ('.docx', '.doc'):
        data = UnstructuredWordDocumentLoader(str(data_path), mode='elements').load()
    elif '.txt' == suffix:
        data = TextLoader(str(data_path),encoding='utf-8')
    elif '.pdf' == suffix:
        data = PyMuPDFLoader(str(data_path))
    elif '.ipynb' == suffix:
        data = NotebookLoader(str(data_path), include_outputs=True, max_output_length=20, remove_newline=True)
    else:
        raise NotImplementedError
    return data


def Loading_files(folder_path):
    filePaths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".ipynb"):
                file_path = os.path.join(root, file)
                file_path = Path(file_path)
                filePaths.append(file_path)
    return filePaths

# print("loading data...")
# folder_path = "owngpt\knowledgeBase"
# filePaths = Loading_files(folder_path)
# loaders = [read_data(file_path) for file_path in filePaths]
# all_documents = []

# for loader in loaders:
#     print("Loading raw document..." + loader.file_path)
#     raw_documents = loader.load()

#     print("Splitting text...")
#     # text_splitter = CharacterTextSplitter(
#     #     separator="\n\n",
#     #     chunk_size=1000,
#     #     chunk_overlap=100,
#     #     length_function=len,
#     # )
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=700,
#         chunk_overlap=100,
#         length_function=len,
#         add_start_index=True,
#     )
#     documents = text_splitter.split_documents(raw_documents)
#     all_documents.extend(documents)

# print("Creating vectorstore...")
# embeddings = OpenAIEmbeddings()
# # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vectorstore = FAISS.from_documents(all_documents,embeddings)


# with open("vectorstore.pkl","wb") as f:
#     pickle.dump(vectorstore,f)








