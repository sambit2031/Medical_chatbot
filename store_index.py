from dotenv import load_dotenv
from src.helper import doc_loader,filter_to_minimal_docs,text_split,download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACEHUB_API_TOKEN

extracted_data=doc_loader(Data='Data/')
filtered_data=filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filtered_data)

embeddings=download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws",region="us-east-1"))
index=pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
