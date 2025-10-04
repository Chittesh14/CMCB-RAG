from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings



#extract text from pdf files
def loadpdffiles(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents=loader.load()
    return documents



def filter_doc(docs:List[Document]) -> List[Document]:
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs


def text_chunker(filtered_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
    )
    texts=text_splitter.split_documents(filtered_docs)
    return texts


def download_embeddings():
    model_name="abhinand/MedEmbed-base-v0.1"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings