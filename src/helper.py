from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings


def doc_loader(Data):
   loader=DirectoryLoader(
        Data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
   return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """ 
    Given List of Document objects,return a new Document object with only 'source' as its metadata and the original page_content
    """
    minimal_docs: List[Document]=[]

    for doc in docs:
        minimal_docs.append(Document(
            metadata= {"source": doc.metadata.get("source")},
            page_content=doc.page_content
        ))
    return minimal_docs


def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text=text_splitter.split_documents(minimal_docs)
    return text


def download_embeddings():
    """Downloads and returns Huggingface embedding model"""
    Embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Embeddings

