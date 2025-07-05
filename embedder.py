#%% [markdown]
# # Embedder Script for RAG System
# This script processes PDF documents, extracts text, chunks it, generates embeddings using Vertex AI, and stores the data in ChromaDB.
# **Execute this script first to populate your knowledge base.**

#%% [markdown]
# ### 1. Imports and Configuration
# Adjust `PDF_DIRECTORY` and `CURRENT_TOPIC` for each set of PDFs you want to embed.

#%%
# embedder.py

# --- Imports ---
import os
import vertexai
import time
import chromadb
from dotenv import load_dotenv # Added to ensure .env is loaded here too

# Make sure necessary functions are imported from rag_core
from rag_core import (
    extract_text_from_pdf,
    chunk_text,
    generate_embeddings,
    store_embeddings_in_chroma
)

# --- Configuration ---
PROJECT_ID = "project-embedding-and-rag"
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-005"

# --- PDF Source Directory and TOPIC ---
# Example: Use a subfolder for 'ism' papers, then change to another for 'antennae' papers.
PDF_DIRECTORY = "./data/gen_ai_whitepapers"   # <--- SET PDF FOLDER PATH (e.g., './data/ism_papers')
CURRENT_TOPIC = "gen_ai"

# --- ChromaDB Output (Keep these the same for all runs) ---
CHROMA_PERSIST_DIR = "./chroma_db_gen_ai" # Use a new/clear directory for the combined DB
CHROMA_COLLECTION_NAME = "gen_ai_whitepapers"      # Use a suitable name for the combined collection

#%% [markdown]
# ### 2. Authentication and Vertex AI Initialization
# This cell handles setting up your GCP credentials and initializing the Vertex AI client.

#%%
# --- Set Authentication ---
load_dotenv() # Load environment variables from .env file
key_path = os.getenv("GCP_KEY_PATH")
if key_path and os.path.exists(key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    print(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {os.path.abspath(key_path)}")
else:
    print("GCP_KEY_PATH not found or invalid in .env. Attempting Application Default Credentials (ADC)...")
    # If key_path is invalid, ensure GOOGLE_APPLICATION_CREDENTIALS is not set to a bad path
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]


# --- Initialize Vertex AI ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("Vertex AI Initialized Successfully.")
except Exception as e:
    print(f"Failed during Vertex AI initialization: {e}")
    print("Please ensure your GCP_KEY_PATH in .env is correct, or ADC is configured.")
    # In a cell, you might want to stop execution here, or just let the error propagate.
    # For a script run, `exit()` is common. For interactive cells, it just prints.

#%% [markdown]
# ### 3. ChromaDB Initialization Check
# Ensures the ChromaDB collection is ready.

#%%
# --- Initialize ChromaDB Client (just to check/create collection once) ---
try:
    chroma_client_check = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    initial_collection = chroma_client_check.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    print(f"Ensured ChromaDB collection '{CHROMA_COLLECTION_NAME}' exists. Initial count: {initial_collection.count()}")
    del initial_collection # Release reference, store function will reconnect
    del chroma_client_check
except Exception as e:
    print(f"Failed to initialize/check ChromaDB client or collection: {e}")
    # Consider raising an exception or providing user guidance here

#%% [markdown]
# ### 4. Find PDF Files and Process
# This is the main loop that processes each PDF in the specified directory.

#%%
# --- Main Execution for Embedding ---
print(f"--- Starting PDF Embedding Process for Topic: {CURRENT_TOPIC} ---")
print(f"Looking for PDFs in: {PDF_DIRECTORY}")
print(f"Storing in ChromaDB Dir: {CHROMA_PERSIST_DIR}, Collection: {CHROMA_COLLECTION_NAME}")

# --- Find PDF files ---
pdf_files = []
try:
    if not os.path.exists(PDF_DIRECTORY):
        print(f"Error: PDF directory not found at {PDF_DIRECTORY}")
    else:
        pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf") and not f.startswith("._")]
except Exception as e:
    print(f"Error finding PDF files: {e}")
print(f"Found {len(pdf_files)} PDF file(s) for topic '{CURRENT_TOPIC}'.")

# --- Loop through and process each PDF ---
total_chunks_processed = 0
processed_files = 0
failed_files = []

for pdf_filename in pdf_files:
    pdf_full_path = os.path.join(PDF_DIRECTORY, pdf_filename)
    print(f"\n--- Processing: {pdf_filename} ---")

    try:
        extracted_text = extract_text_from_pdf(pdf_full_path)
        if not extracted_text:
            print(f"   Skipping {pdf_filename}: No text extracted.")
            failed_files.append(pdf_filename)
            continue

        text_chunks = chunk_text(extracted_text, max_chunk_size=1000, overlap=100)
        if not text_chunks:
            print(f"   Skipping {pdf_filename}: No chunks created.")
            failed_files.append(pdf_filename)
            continue

        embedding_vectors = generate_embeddings(text_chunks, model_name=MODEL_NAME, batch_size=20, api_wait_time=0.5)
        if not embedding_vectors or len(embedding_vectors) != len(text_chunks):
            print(f"   Skipping {pdf_filename}: Failed to generate embeddings for all chunks.")
            failed_files.append(pdf_filename)
            continue

        print(f"   Generated {len(embedding_vectors)} embeddings for {pdf_filename}.")

        collection_ref = store_embeddings_in_chroma(
            text_chunks=text_chunks,
            embedding_vectors=embedding_vectors,
            source_identifier=pdf_filename,
            topic=CURRENT_TOPIC,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )

        if collection_ref:
            total_chunks_processed += len(text_chunks)
            processed_files += 1
            print(f"   Successfully stored embeddings for {pdf_filename}.")
        else:
            print(f"   Failed to store embeddings for {pdf_filename}.")
            failed_files.append(pdf_filename)

    except Exception as file_e:
        print(f"   An unexpected error occurred processing {pdf_filename}: {file_e}")
        failed_files.append(pdf_filename)

#%% [markdown]
# ### 5. Summary
# Final statistics after processing all PDFs for the current topic.

#%%
# --- Summary ---
print("\n--- Embedding Process Summary ---")
print(f"Finished processing for Topic: {CURRENT_TOPIC}")
print(f"Successfully processed {processed_files} out of {len(pdf_files)} PDF files.")
print(f"Total chunks added to ChromaDB in this run (approx): {total_chunks_processed}")
if failed_files:
    print(f"Failed to process the following files: {failed_files}")
else:
    print("All files processed successfully!")

try:
    final_collection = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR).get_collection(name=CHROMA_COLLECTION_NAME)
    print(f"Final total items in collection '{CHROMA_COLLECTION_NAME}': {final_collection.count()}")
    print(f"ChromaDB database saved to: {os.path.abspath(CHROMA_PERSIST_DIR)}")
except Exception as e:
    print(f"Could not get final ChromaDB collection count: {e}")
