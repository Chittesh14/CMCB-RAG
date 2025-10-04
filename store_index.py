from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone import Pinecone
from dotenv import load_dotenv
from src.helper import loadpdffiles,filter_doc,text_chunker,download_embeddings
import os



#inserting data in pine cone
load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

extracted_data=loadpdffiles("Data/")
filtered_docs=filter_doc(extracted_data)
chunked_text=text_chunker(filtered_docs)
embedding=download_embeddings()

pinecone_api_key=PINECONE_API_KEY


pc=Pinecone(api_key=pinecone_api_key)

#creating index in pinecone DB
index_name="medical-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec("aws",region="us-east-1")
    )
index=pc.Index(index_name)

#insert into pinecone DB

batch_size = 200  # safe batch size
for i in range(0, len(chunked_text), batch_size):
    batch_docs = chunked_text[i:i+batch_size]
    PineconeVectorStore.from_documents(
        documents=batch_docs,
        embedding=embedding,
        index_name=index_name
    )
    print(f"Upserted batch {i} to {i+len(batch_docs)}")