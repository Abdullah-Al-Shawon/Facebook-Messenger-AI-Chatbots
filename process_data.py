import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import pinecone
from tqdm import tqdm

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit()
genai.configure(api_key=GEMINI_API_KEY)

pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name not in pc.list_indexes().names():
    print(f"Error: Index '{index_name}' not found.")
    print("Please create the index in the Pinecone dashboard.")
    exit()

index = pc.Index(index_name)

def process_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        chunks = [chunk for chunk in full_text.strip().split('\n\n') if chunk.strip()]
        print(f"Found {len(chunks)} non-empty chunks in '{file_path}'.")
        return chunks
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        time.sleep(5)
        return None

def upload_to_pinecone(chunks, batch_size=50):
    if not chunks:
        print("No data to upload.")
        return

    print(f"Starting upload of {len(chunks)} chunks...")
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Pinecone"):
        batch_chunks = chunks[i:i+batch_size]
        
        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            embedding = get_embedding(chunk)
            if embedding:
                vector_id = f"chunk_{i+j}"
                metadata = {'text': chunk}
                vectors_to_upsert.append((vector_id, embedding, metadata))
        
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)

    print("\nData upload to Pinecone completed successfully!")

if __name__ == "__main__":
    file_to_process = "chat.txt" 
    
    text_chunks = process_text_file(file_to_process)
    
    if text_chunks:
        upload_to_pinecone(text_chunks)
        
        stats = index.describe_index_stats()
        print("\nCurrent Pinecone index stats:")
        print(stats)
