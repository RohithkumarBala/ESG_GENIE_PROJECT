'''
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def create_vector_db(pdf_path):
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    # 2. Split into chunks for the AI to read
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # 3. Create the Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Store in ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Success! Your 'chroma_db' folder is ready.")

if __name__ == "__main__":
    # Ensure you have an ESG-related PDF named 'data.pdf' in the folder
    create_vector_db("data.pdf")
'''
'''
# updated ingest.py to handle Google API rate limits by processing in batches and adding delays between requests.
import os
import time  # Add this import
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def create_vector_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    # Increase chunk size slightly to reduce the total number of requests
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # We will process in smaller batches to avoid the 429 error
    batch_size = 50 
    vector_db = None

    print(f"Starting ingestion of {len(chunks)} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        if vector_db is None:
            vector_db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vector_db.add_documents(batch)
        
        print(f"Processed {i + len(batch)} / {len(chunks)} chunks...")
        time.sleep(2)  # Pause for 2 seconds between batches

    print("✅ Success! Database created in 'chroma_db' folder.")

if __name__ == "__main__":
    create_vector_db("data.pdf")

'''
# Final version of ingest.py with enhanced error handling for Google API rate limits, including retries and longer wait times after hitting the limit.
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
# from google.api_core import exceptions

load_dotenv()

def create_vector_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    # 1. Larger chunks = fewer API calls
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 2. Smaller batch size to stay under the 100 limit
    batch_size = 10 
    vector_db = None

    print(f"🚀 Starting ingestion of {len(chunks)} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        success = False
        retries = 0
        
        while not success and retries < 5:
            try:
                if vector_db is None:
                    vector_db = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )
                else:
                    vector_db.add_documents(batch)
                
                success = True
                print(f"✅ Processed {i + len(batch)} / {len(chunks)} chunks...")
                # Small pause between successful batches
                time.sleep(5) 
                
            except Exception as e:
                if "429" in str(e):
                    retries += 1
                    wait_time = 35  # Wait 35 seconds as suggested by error
                    print(f"⚠️ Rate limit hit. Waiting {wait_time} seconds before retry {retries}/5...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ An unexpected error occurred: {e}")
                    break

    print("\n🎉 MISSION ACCOMPLISHED! Your 'chroma_db' is fully built.")

if __name__ == "__main__":
    # IMPORTANT: Delete your old 'chroma_db' folder manually before running this!
    create_vector_db("data.pdf")