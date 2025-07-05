# rag_app.py

import streamlit as st
import chromadb
import vertexai
import os
from dotenv import load_dotenv

# --- Import core RAG functions ---
try:
    from rag_core import query_rag_system
except ImportError:
    st.error("Failed to import from rag_core.py. Make sure rag_core.py is in the same directory.")
    st.stop()

# --- Configuration ---
load_dotenv()
PROJECT_ID = "project-embedding-and-rag"
LOCATION = "us-central1"
EMBEDDING_MODEL_NAME = "text-embedding-005"
LLM_MODEL_NAME = "gemini-1.5-pro-002" # Using Pro for potentially better synthesis

# --- Point to the CONSOLIDATED database ---
CHROMA_PERSIST_DIR = "./chroma_db_gen_ai" # Updated path
CHROMA_COLLECTION_NAME = "gen_ai_whitepapers" # Updated name

# --- Store Model Token Limits (Based on Documentation) ---
MODEL_TOKEN_LIMITS = {
    "text-embedding-005": {"input_per_chunk": 2048},
    "text-embedding-004": {"input_per_chunk": 2048},
    "gemini-2.0-flash-001": {"input_total": 1048576, "output": 8192},
    "gemini-1.5-pro-001": {"input_total": 1048576, "output": 8192}, # Using 1M context limit
    "gemini-1.5-pro-002": {"input_total": 1048576, "output": 8192}, # Using 1M context limit
    # Add other models here as you use them
}

# === Authentication & Initialization (Cached) ===

# Set GOOGLE_APPLICATION_CREDENTIALS from .env for Vertex AI libs
@st.cache_data(show_spinner=False) # Cache the result of setting this up
def configure_auth():
    key_path = os.getenv("GCP_KEY_PATH")
    if key_path and os.path.exists(key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        print(f"Using GOOGLE_APPLICATION_CREDENTIALS from .env: {os.path.abspath(key_path)}")
        return True
    elif key_path:
        st.error(f"Key file specified in .env not found at {key_path}. Using ADC.")
        # Unset it if path is invalid to ensure fallback to ADC works cleanly
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        return False # Indicate key file failed
    else:
        print("GCP_KEY_PATH not found in .env. Attempting Application Default Credentials (ADC)...")
        # Assume ADC will be picked up automatically if env var isn't set
        return False # Indicate key file wasn't used

# Initialize Vertex AI (runs only once per session thanks to caching)
@st.cache_resource(show_spinner=False)
def init_vertexai_cached(project, location):
    print("Initializing Vertex AI...")
    try:
        vertexai.init(project=project, location=location)
        print("Vertex AI Initialized.")
        return True
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI: {e}")
        st.error("Please ensure authentication is configured correctly (check .env or ADC).")
        return False

# Connect to ChromaDB (runs only once per session thanks to caching)
# MODIFIED: Now returns both the collection and a list of unique topics found in its metadata
@st.cache_resource(show_spinner=False)
def connect_chroma_cached(persist_dir, collection_name):
    print(f"Connecting to ChromaDB at {persist_dir}...")
    collection = None
    topics = [] # Initialize topics list
    try:
        if not os.path.exists(persist_dir):
             st.error(f"ChromaDB directory not found at {persist_dir}. Did embedder.py run successfully?")
             return None, [] # Return None for collection and empty list for topics
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_collection(name=collection_name)
        count = collection.count() # Get count after getting collection
        print(f"Connected to collection '{collection_name}' with {count} items.")
        if count == 0:
            st.warning(f"ChromaDB collection '{collection_name}' is empty. Did embedder.py populate it?")
            return collection, [] # Return empty list if collection is empty

        # Dynamically get unique topics from metadata
        # This fetches all IDs, then gets metadata for those IDs.
        # For very large collections, this might be slow, but for typical RAG projects, it's fine.
        all_ids = collection.get()['ids']
        if all_ids: # Only proceed if there are IDs
            all_metadatas = collection.get(ids=all_ids, include=['metadatas'])['metadatas']
            unique_topics = set()
            for meta in all_metadatas:
                if 'topic' in meta:
                    unique_topics.add(meta['topic'])
            topics = sorted(list(unique_topics)) # Sort for consistent display
        print(f"Found topics in DB: {topics}")

        return collection, topics
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB collection '{collection_name}' at {persist_dir}: {e}")
        return None, [] # Return None for collection and empty list for topics

# --- Initialize ---
auth_configured = configure_auth()
vertexai_ready = init_vertexai_cached(PROJECT_ID, LOCATION)
# MODIFIED: Capture both the collection and the dynamically loaded topics
chroma_collection, available_topics_from_db = connect_chroma_cached(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("💡 Document Research Assistant") # Updated title

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Assistant Configuration")

    # Task Selection - Updated for more general tasks
    available_tasks = [
        "Ask a Question",            # General Q&A
        "Find Supporting Evidence",  # For claims
        "Help Draft Content"         # General drafting
    ]
    selected_task = st.selectbox(
        "Select Task:",
        options=available_tasks,
        key="task_selector"
    )

    # Topic Selection (conditional) - Update condition
    # MODIFIED: Use dynamically loaded topics from the database
    if not available_topics_from_db:
        st.warning("No topics found in the database. Please run embedder.py to populate it.")
        display_topics_for_multiselect = [] # No topics to select if DB is empty
        default_selected_topics = []
    else:
        display_topics_for_multiselect = available_topics_from_db
        default_selected_topics = available_topics_from_db # Default to all found topics

    selected_topics = []
    # Show topic selector for all tasks that involve searching the document database
    # Since all new tasks will use the document database, this condition is simplified
    if display_topics_for_multiselect: # Only show if there are topics to select
        selected_topics = st.multiselect(
            "Filter Document Topics (Optional):", # More general label
            options=display_topics_for_multiselect,
            default=default_selected_topics,
            key="topic_selector"
        )
        # Ensure default selection if user deselects all, and only if there were topics to begin with
        if not selected_topics and default_selected_topics:
            selected_topics = default_selected_topics # Keep default if empty
            st.caption("No specific topic selected; searching all available topics.")
        elif not selected_topics and not default_selected_topics:
            st.caption("No topics available for selection.")
    else:
        st.caption("No topics available for filtering. Please embed documents first.")


    # Context Amount Slider (Keep as is)
    n_results = st.slider(
        "Number of context chunks to retrieve:", 1, 500, 10, 1, key="n_results_slider"
    )

    st.markdown("---")
    # Display Model Info & Limits
    st.subheader("Models & Limits")
    st.markdown("---")
    st.info(f"""
            **Database Info:**
            * Collection: `{CHROMA_COLLECTION_NAME}`
            * Items: `{chroma_collection.count() if chroma_collection else 'N/A'}`
            * Directory: `{CHROMA_PERSIST_DIR}`
            """)


# --- Main Interaction Area ---
if vertexai_ready and chroma_collection:
    st.success(f"Connected to Vector DB '{CHROMA_COLLECTION_NAME}'. Ready for questions.")

    col1, col2 = st.columns([2,1])

    with col1:
        # --- Use dynamic label (Optional but helpful) ---
        input_label = "Enter your query or request based on the selected task..." # Default general label
        if selected_task == "Ask a Question":
            input_label = "Enter your question about the documents:"
        elif selected_task == "Find Supporting Evidence":
            input_label = "Enter the claim or statement you want evidence for:"
        elif selected_task == "Help Draft Content":
            input_label = "Describe the content you want help drafting (e.g., 'Summary of X', 'Introduction to Y'):"

        user_query = st.text_area(input_label, height=100, key="query_input") # Use dynamic label
        submit_button = st.button("Get Answer ✨")

        # Display results area
        st.markdown("--- \n ### Answer:")
        answer_placeholder = st.empty() # Create a placeholder for the answer/status
        token_info_placeholder = st.empty() # Create placeholder for token info


    # Display Model Info & Limits in Sidebar/Column 2
    with col2:
        st.subheader("⚙️ Configuration")
        embedding_limits = MODEL_TOKEN_LIMITS.get(EMBEDDING_MODEL_NAME, {})
        llm_limits = MODEL_TOKEN_LIMITS.get(LLM_MODEL_NAME, {})
        st.markdown(f"""
        **Embedding Model:** `{EMBEDDING_MODEL_NAME}`
        * Max Input (per chunk): `{embedding_limits.get('input_per_chunk', 'N/A')}` tokens
        """)
        st.markdown(f"""
        **LLM:** `{LLM_MODEL_NAME}`
        * Approx Max Total Input: `{llm_limits.get('input_total', 'N/A')}` tokens
        * Max Output: `{llm_limits.get('output', 'N/A')}` tokens
        """)
        st.caption("Token limits based on documentation and may vary slightly.")
        st.markdown("---")
        st.info(f"""
        **Database Info:**
        * Collection: `{CHROMA_COLLECTION_NAME}`
        * Items: `{chroma_collection.count() if chroma_collection else 'N/A'}`
        * Directory: `{CHROMA_PERSIST_DIR}`
        """)


    # Handle submission
    if submit_button and user_query:
        answer_placeholder.info("🧠 Thinking... Please wait.")
        token_info_placeholder.empty()

        with st.spinner("Embedding query, searching documents, asking LLM..."):
            # --- UPDATE THIS CALL ---
            result = query_rag_system(
                query_text=user_query,
                chroma_collection=chroma_collection, # Get collection directly
                embedding_model_name=EMBEDDING_MODEL_NAME,
                llm_model_name=LLM_MODEL_NAME,
                task=selected_task,              # <-- Pass selected task
                selected_topics=selected_topics, # <-- Pass selected topics
                n_results=n_results,             # <-- Use n_results from sidebar slider
                # chat_history=history_for_prompt # <-- Uncomment if using chat history later
            )
            # ---

        # Update placeholders with result
        if 'error' in result:
            answer_placeholder.error(result['error'])
        elif 'answer' in result:
            answer_placeholder.markdown(result['answer'])
            token_info_placeholder.caption(
                f"Token Usage (LLM Call): Prompt ≈ {result.get('prompt_tokens', 'N/A')}, "
                f"Completion = {result.get('completion_tokens', 'N/A')}"
            )
        else:
             answer_placeholder.error("An unexpected error occurred retrieving the answer.")


    elif submit_button and not user_query:
        answer_placeholder.warning("Please enter a question.")


elif not vertexai_ready:
     st.error("🔴 Vertex AI failed to initialize. Cannot process queries. Check authentication and project setup.")
elif not chroma_collection:
     st.error("🔴 Failed to connect to the ChromaDB Vector Database. Cannot process queries. Check path and if embedder script ran.")
