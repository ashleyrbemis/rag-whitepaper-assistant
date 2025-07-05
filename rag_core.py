import PyPDF2
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, Tool, grounding
import traceback
import re
import time
import chromadb
import os

MODEL_NAME = "text-embedding-005"      # Default embedding model

# --- Step 1: Extract Text from PDF ---
def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF file and extracts text content from all pages.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: The concatenated text content from the PDF, or None if an error occurs.
    """
    print(f"Extracting text from: {pdf_path}")
    full_text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            print(f"Found {num_pages} pages.")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    # Basic cleanup (optional): replace multiple newlines/spaces
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    full_text += page_text + "\n" # Add newline between pages

        print("Text extraction completed.")
        return full_text.strip() # Remove leading/trailing whitespace
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return None

# --- Step 2: Chunk Text ---
def chunk_text(text, max_chunk_size=1000, overlap=100):
    """
    Splits a long text into smaller overlapping chunks.

    Args:
        text (str): The input text to be chunked.
        max_chunk_size (int): The approximate maximum number of characters per chunk.
        overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    print(f"Chunking text (max_chunk_size={max_chunk_size}, overlap={overlap})...")
    if not text:
        return []

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = min(start_index + max_chunk_size, len(text))
        chunks.append(text[start_index:end_index])
        start_index += max_chunk_size - overlap
        # Ensure overlap doesn't push start_index backward
        if start_index < 0 or (len(chunks) > 1 and start_index <= (end_index - max_chunk_size)):
            start_index = end_index - overlap # Adjust if step is too small
        if start_index >= len(text): # Prevent infinite loop on very small texts
            break

    print(f"Created {len(chunks)} chunks.")
    return chunks

# --- Step 3: Generate Text Embeddings (with Batching) ---
def generate_embeddings(text_chunks, model_name=MODEL_NAME, batch_size=5, api_wait_time=1.0):
    """
    Generates embeddings for a list of text chunks using Vertex AI, processing in batches.

    Args:
        text_chunks (list[str]): A list of text strings to embed.
        model_name (str): The name of the embedding model to use.
        batch_size (int): The number of chunks to process in each API call.
                           Vertex AI embedding APIs often have limits (e.g., 250 texts per request).
                           Adjust based on model limits and observed performance.
        api_wait_time (float): Seconds to wait between batch API calls to avoid hitting rate limits.

    Returns:
        list: A list of embedding vectors (each vector is a list of floats),
              or an empty list if an error occurs or no chunks are provided.
              Errors in specific batches are printed but skipped.
    """
    print(f"Generating embeddings using model: {model_name} with batch_size={batch_size}...")
    if not text_chunks:
        print("No text chunks provided for embedding.")
        return []

    all_embedding_vectors = []
    try:
        # Initialize the model once
        model = TextEmbeddingModel.from_pretrained(model_name)

        # Process chunks in batches
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(text_chunks) + batch_size - 1) // batch_size}...")

            try:
                # Get embeddings for the current batch
                embeddings_response = model.get_embeddings(batch)
                # Extract the embedding values
                batch_embedding_vectors = [embedding.values for embedding in embeddings_response]
                all_embedding_vectors.extend(batch_embedding_vectors)
                print(f"Successfully generated {len(batch_embedding_vectors)} embeddings for this batch.")

                # Optional: Wait a bit between batches to respect potential rate limits
                if api_wait_time > 0 and i + batch_size < len(text_chunks):
                    time.sleep(api_wait_time)

            except Exception as batch_e:
                # Log the error for the specific batch but continue if possible
                print(f"Error processing batch starting at index {i}: {batch_e}")
                # Optionally add None placeholders for failed chunks in the batch
                # all_embedding_vectors.extend([None] * len(batch))
                # Or simply skip the batch results

        print(f"Finished generating embeddings. Total vectors generated: {len(all_embedding_vectors)}")
        return all_embedding_vectors

    except Exception as e:
        # Handle potential model initialization errors or other critical failures
        print(f"A critical error occurred during embedding generation setup: {e}")
        return [] # Return empty list on critical failure

# === Step 4: Store Embeddings in ChromaDB ===
def store_embeddings_in_chroma(
    text_chunks,
    embedding_vectors,
    source_identifier,
    topic, # <-- Add topic parameter
    persist_directory,
    collection_name
    ):
    """
    Stores text chunks and their corresponding embeddings in a ChromaDB collection,
    including source identifier and topic in the metadata.

    Args:
        text_chunks (list[str]): The list of original text chunks from one document.
        embedding_vectors (list[list[float]]): The list of embedding vectors for those chunks.
        source_identifier (str): An identifier for the source document (e.g., filename).
        topic (str): A tag identifying the topic or source category (e.g., 'ism', 'antennae').
        persist_directory (str): The directory to store ChromaDB data.
        collection_name (str): The name of the collection to create or use.

    Returns:
        chromadb.Collection: The ChromaDB collection object, or None on failure.
    """
    if len(text_chunks) != len(embedding_vectors):
        print(f"Error storing {source_identifier}: Chunk/vector count mismatch.")
        return None
    if not embedding_vectors:
        print(f"Error storing {source_identifier}: No vectors.")
        return None

    print(f"\nStoring embeddings for '{source_identifier}' (Topic: {topic}) in ChromaDB...")

    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        # Consider adding settings=chromadb.Settings(anonymized_telemetry=False) if desired
        collection = chroma_client.get_or_create_collection(name=collection_name)

        num_chunks = len(text_chunks)
        ids = [f"{source_identifier}_chunk_{i}" for i in range(num_chunks)] # Unique IDs

        # --- Include topic in metadata ---
        metadatas = [
            {'source': source_identifier, 'topic': topic, 'chunk_index': i}
            for i in range(num_chunks)
        ]
        # ---

        batch_size = 500 # Adjust as needed
        for i in range(0, num_chunks, batch_size):
            batch_end = min(i + batch_size, num_chunks)
            print(f"  Adding batch {i // batch_size + 1}/{(num_chunks + batch_size - 1) // batch_size} "
                  f"(indices {i} to {batch_end-1}) for '{source_identifier}'...")

            collection.add(
                embeddings=embedding_vectors[i:batch_end],
                documents=text_chunks[i:batch_end],
                metadatas=metadatas[i:batch_end], # Pass updated metadata
                ids=ids[i:batch_end]
            )
        print(f"  Batches for '{source_identifier}' added successfully.")
        return collection

    except Exception as e:
        print(f"An error occurred storing embeddings for '{source_identifier}' in ChromaDB: {e}")
        return None


# === Step 5: Query the RAG System ===
def query_rag_system(
    query_text,
    chroma_collection,
    embedding_model_name,
    llm_model_name,
    task="Ask a Question", # Updated default task
    selected_topics=None,
    n_results=5,
    chat_history=None
    ):
    """
    Queries the RAG system, returns answer and token usage metadata.

    Args:
        query_text (str): The user's question.
        chroma_collection (chromadb.Collection): The ChromaDB collection containing embeddings.
        embedding_model_name (str): The name of the Vertex AI model used for embeddings.
        llm_model_name (str): The name of the Vertex AI generative model (e.g., Gemini).
        task (str): The type of task requested (e.g., "Ask a Question", "Find Supporting Evidence", "Help Draft Content").
        selected_topics (list[str], optional): List of topics to filter the document search by. Defaults to None, meaning all topics.
        n_results (int): The number of relevant chunks to retrieve from ChromaDB.

    Returns:
        dict: A dictionary containing 'answer', 'prompt_tokens', 'completion_tokens',
              or an 'error' key if something failed.
    """
    if not query_text:
        return {'error': "Error: No query text provided."}
    if not chroma_collection:
         return {'error': "Error: ChromaDB collection is not available."}

    print(f"\n--- Querying RAG Assistant ---")
    print(f"Task: {task}")
    print(f"User Query/Input: {query_text}")
    if selected_topics:
        print(f"Selected Topics: {selected_topics}")

    try:
        # --- Determine ChromaDB Filter and Search Query ---
        chroma_where_filter = None
        search_query = query_text
        
        # Use selected_topics directly for filtering
        if selected_topics:
            if len(selected_topics) == 1:
                chroma_where_filter = {"topic": selected_topics[0]}
            elif len(selected_topics) > 1:
                chroma_where_filter = {"topic": {"$in": selected_topics}}
            print(f"Filtering ChromaDB query for topic(s): {selected_topics}")
        else:
            print("No specific topic filter applied; searching across all topics.")

        # Determine target_topics_string for display in prompts
        if selected_topics:
            target_topics_string = ", ".join(selected_topics)
        else:
            target_topics_string = "All Available Documents" # More general

        print(f"DEBUG: target_topics_string set to: '{target_topics_string}'")

        # 1. Embed the User Query
        print(f"Embedding search query: '{search_query}'...")
        query_embedding_list = generate_embeddings([search_query], model_name=embedding_model_name)
        if not query_embedding_list or not query_embedding_list[0]:
            return "Error: Failed to generate embedding for the query."
        query_embedding = query_embedding_list[0]
        print(f"Query embedding generated.")

        # 2. Query ChromaDB with Filter
        actual_n_results = n_results
        if task in ["Find Supporting Evidence", "Help Draft Content"]: # Updated task name
             actual_n_results = max(n_results, 10) # Retrieve at least 10 chunks for these tasks
             print(f"Retrieving increased number of chunks ({actual_n_results}) for task '{task}'.")

        print(f"Querying ChromaDB (n_results={actual_n_results}, filter={chroma_where_filter})...")
        results = chroma_collection.query(
             query_embeddings=[query_embedding],
             n_results=actual_n_results, # Use potentially adjusted n_results
             where=chroma_where_filter,
             include=['documents', 'metadatas']
        )

        # Check if results are valid and contain necessary parts
        if not results or not results.get('ids') or not results['ids'][0]:
             return "Could not find relevant documents in the database for this query."

        # Handle cases where fewer results than n_results are returned
        num_found = len(results['ids'][0]) if results and results.get('ids') and results['ids'][0] else 0
        context_string = "No specific context was found in the documents for the current query and filter."
        if num_found > 0:
            print(f"Retrieved {num_found} context chunks.")
            # Format context string including sources
            context_items = []
            for i in range(num_found):
                 doc = results['documents'][0][i]
                 meta = results['metadatas'][0][i]
                 source = meta.get('source', 'Unknown Source')
                 context_items.append(f"Source: {source}\nContent:\n{doc}\n---")
            context_string = "\n\n".join(context_items)
        else:
             print("No relevant documents found in DB for this query and filter.")

        # 3. Construct the Prompt for the LLM
        history_string = ""

        if chat_history:
            print("DEBUG: Entering chat_history block...")
            limited_history = chat_history[-6:]
            print(f"DEBUG: limited_history assigned, length {len(limited_history)}")
            history_parts = []
            for i, msg in enumerate(limited_history):
                print(f"DEBUG: Processing history message {i}")
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                history_parts.append(f"{role}: {content}")

            history_string = "\n".join(history_parts) + "\n\n"
            print("DEBUG: Finished processing chat_history block.")
        else:
            print("DEBUG: Skipped chat_history block (history is empty or None).")


        if task == "Ask a Question": # Updated task name
             prompt = f"""
You are an AI assistant answering questions based ONLY on the provided context excerpts from documents (topics: {target_topics_string}), the conversation history, and potentially supplemental, verified information from an external Google Search. You may use external Google Search via available tools to verify information or find supplemental details if the context is insufficient, but clearly indicate and cite any information obtained from web searches.
Each context block is preceded by its source identifier (e.g., 'Source: document.pdf').

CONVERSATION HISTORY:
{history_string}
CONTEXT FROM DOCUMENTS:
---
{context_string}
---

LATEST USER QUESTION:
{query_text}

ANSWER:
Based *only* on the conversation history and the provided context from documents:
- Provide a detailed and comprehensive answer to the **latest user question**.
- Refer to the conversation history if relevant.
- Synthesize information from the relevant sources as needed.
- **Cite the source identifier(s) (e.g., "[Source: document.pdf]")** for the specific information used from the context. Cite throughout your response where appropriate.
- If the answer cannot be found in the context, state that clearly.
"""
        elif task == "Find Supporting Evidence":
             prompt = f"""
You are an AI assistant evaluating evidence for a user's claim based ONLY on provided context excerpts from documents.
Each context block is preceded by its source identifier (e.g., 'Source: document.pdf').

USER'S CLAIM:
"{query_text}"

CONTEXT FROM DOCUMENTS (Topics: {target_topics_string}):
---
{context_string}
---

INSTRUCTIONS:
Based *only* on the provided context and the user's claim:
1.  Carefully review each context excerpt.
2.  Identify excerpts that **directly support** the user's claim.
3.  Identify excerpts that **directly contradict** the user's claim.
4.  Identify excerpts that are **related** to the claim but neither directly support nor contradict it.
5.  Summarize your findings. For each piece of supporting or contradictory evidence identified, quote the relevant sentence(s) from the context and **cite the source identifier** (e.g., "[Source: document.pdf]").
6.  If no relevant supporting or contradictory evidence is found in the context, state that clearly. Do not add external information.

EVALUATION:
"""
        elif task == "Help Draft Content": # Updated task name
            prompt = f"""
You are an expert assistant for drafting content based **primarily** on provided context excerpts, ensuring the output flows well, uses **original phrasing**, and remains **factually grounded**. You may use an external Google Search via available tools to verify information or find supplemental details if the context is insufficient, but clearly **indicate and cite any information obtained from web searches.**

Now, follow these instructions carefully:

CONTEXT FROM DOCUMENTS (Topics: {target_topics_string}):
---
{context_string}
---

INSTRUCTIONS:
Based on the provided context AND mimicking a formal, objective, and scholarly writing style suitable for technical documents:
1.  Draft the specified content for the topic described.
2.  Structure the content logically. Ensure smooth transitions.
3.  **Synthesize information:** Combine related ideas from different context sources where appropriate.
4.  **Generate unique text:** Rephrase the information from the context **in your own words**. **Do NOT directly copy sentences or long phrases from the provided CONTEXT.** Ensure the draft avoids plagiarism while accurately representing the information in the context.
5.  **Grounding:** All specific claims, findings, data points, and core ideas MUST originate from the provided CONTEXT or other verifiable sources. You may use general language capabilities for transitions or minor rephrasing for flow, but introduce no external facts that are not verified.
6.  **Citation:** Cite the source identifier (e.g., "[Source: document.pdf]") for the context excerpt(s) where the original information or idea was found, even after rephrasing. Cite appropriately throughout the text.
7.  If the context is insufficient, state the limitations clearly.

DRAFT CONTENT:
"""

        else:
            return {'error': f"Unknown task type for prompt generation: {task}"}

        print(f"Constructed prompt for task '{task}'.")


        # 4. Define Grounding Tool & Call LLM
        print(f"Calling LLM ({llm_model_name}) with grounding enabled...")
        llm = GenerativeModel(llm_model_name)

        # --- Define the Google Search Tool ---
        search_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
        # ---

        generation_config = GenerationConfig(
            temperature=0.2, # Keep low temp for factuality
            top_p=0.8,
            #max_output_tokens=2048
        )

        # --- Make the call including the 'tools' argument ---
        response = llm.generate_content(
            prompt,
            generation_config=generation_config,
            tools=[search_tool] # <-- Enable grounding tool
            # safety_settings=...
        )
        print("LLM response received.")

        # 5. Extract Answer and Token Usage
        prompt_tokens = 0
        completion_tokens = 0
        answer = None

        if response and response.candidates:
            answer = response.text
            # Extract usage metadata if available
            try:
                if hasattr(response, 'usage_metadata'): # Check if attribute exists
                    usage = response.usage_metadata
                    prompt_tokens = usage.prompt_token_count
                    completion_tokens = usage.candidates_token_count
                    print(f"Token Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                else:
                     print("Warning: usage_metadata not found in response.")
            except AttributeError:
                print("Warning: Could not access usage_metadata attributes.")
            except Exception as meta_e:
                 print(f"Warning: Error processing usage_metadata: {meta_e}")

            return {
                'answer': answer,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        else:
             print("LLM response was empty or blocked.")
             # Inspect feedback: print(response.prompt_feedback)
             return {'error': "LLM response was empty or blocked."}

    except Exception as e:
        print(f"An error occurred during the RAG query process: {e}")
        # print(traceback.format_exc()) # Uncomment for detailed traceback
        return {'error': f"An error occurred during query: {e}"}
