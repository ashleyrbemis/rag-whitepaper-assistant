# Document Research Assistant (RAG System)

## Project Overview

This project implements a Retrieval Augmented Generation (RAG) system designed to assist with research by querying and drafting content based on a specialized corpus of documents. It leverages Google's Vertex AI for embeddings and large language models (LLMs), with ChromaDB as the vector store for efficient document retrieval. The interactive application is built with Streamlit.

## Features

* **PDF Text Extraction:** Extracts content from PDF documents.

* **Intelligent Text Chunking:** Splits documents into manageable, overlapping chunks optimized for embedding and retrieval.

* **Vertex AI Embeddings:** Utilizes Google's `text-embedding-005` model to generate high-quality vector representations of text.

* **ChromaDB Vector Store:** Stores and efficiently retrieves relevant document chunks based on semantic similarity.

* **Vertex AI Gemini LLM Integration:** Employs the `gemini-1.5-pro-002` model to synthesize answers and draft content, grounded in retrieved information.

* **Google Search Grounding:** (Optional, via LLM Tool) Allows the LLM to verify or supplement information using external web search if context is insufficient.

* **Streamlit User Interface:** Provides an intuitive web application for interactive querying and configuration.

* **Flexible Task Capabilities:** Supports general question answering, finding supporting evidence for claims, and assisting with drafting various types of content.

* **Topic Filtering:** Enables filtering document searches by specific topics (dynamically loaded from your embedded data).

## Project Structure

```
.
├── data/
│   └── gen_ai_whitepapers/    # Directory for Gen AI whitepaper PDF documents (your current data)
├── chroma_db_gen_ai/          # Persistent directory for the Gen AI ChromaDB vector store (your current data)
├── .env                       # Environment variables for API keys and paths (DO NOT COMMIT!)
├── rag_core.py                # Core RAG functionalities: extract, chunk, embed, store, query
├── embedder.py                # Script to process PDFs and populate the ChromaDB
├── rag_app.py                 # Streamlit application for the RAG UI
├── requirements.txt           # Python dependencies for the project
└── README.md                  # Project overview and setup instructions (this file)
```

## Setup and Installation

### Prerequisites

* **Python 3.11:** This project is developed and tested with Python 3.11.

* **Anaconda/Miniconda (Recommended):** For managing virtual environments.

* **Google Cloud Project:** Access to a Google Cloud Project with the Vertex AI API enabled.

* **Service Account Key:** A Google Cloud service account key file with permissions for Vertex AI (e.g., `Vertex AI User`, `Service Usage Consumer`).

### 1. Clone the Repository

```bash
git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
cd your_repository_name
```

*(Replace `your_username/your_repository_name` with your actual GitHub path once uploaded.)*

### 2. Set up Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
conda create --name rag_env python=3.11
conda activate rag_env
```

### 3. Install Dependencies

Install all required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

**Important Note for `protobuf`:** You might encounter an error related to `protobuf` versions (e.g., `TypeError: Descriptors cannot be created directly`). If this occurs, downgrade `protobuf` to a compatible version:

```bash
pip install protobuf==3.20.3
```

### 4. Configure Google Cloud Credentials

Create a `.env` file in the root directory of your project (e.g., where `rag_app.py` and `embedder.py` are located) and add the path to your Google Cloud service account key file:

```
GCP_KEY_PATH="/path/to/your/google-cloud-key.json"
```

*Replace `/path/to/your/google-cloud-key.json` with the actual absolute or relative path to your downloaded JSON key file.*

**Security Note:** Add `.env` to your `.gitignore` file to prevent sensitive credentials from being committed to version control.

### 5. Prepare Your Data

Organize your PDF documents into subdirectories within a `data/` folder (e.g., `data/gen_ai_whitepapers`). The `embedder.py` script expects to be pointed to these directories.

### 6. Populate the Vector Database

Run the `embedder.py` script to process your PDFs and store their embeddings in ChromaDB.

**Example for 'Gen AI' whitepapers:**
Open `embedder.py` and set:
`PDF_DIRECTORY = "./data/gen_ai_whitepapers"`
`CURRENT_TOPIC = "gen_ai"`
`CHROMA_PERSIST_DIR = "./chroma_db_gen_ai"`
`CHROMA_COLLECTION_NAME = "gen_ai_whitepapers"`
Then run:

```bash
python embedder.py
```

### 7. Run the Streamlit Application

Once your database is populated, start the interactive RAG application:

```bash
streamlit run rag_app.py
```

This command will open the application in your web browser.

## How to Contribute

(Optional section for future contributions or if you want to show collaboration skills)

* Fork the repository.

* Create a new branch (`git checkout -b feature/AmazingFeature`).

* Make your changes.

* Commit your changes (`git commit -m 'Add some AmazingFeature'`).

* Push to the branch (`git push origin feature/AmazingFeature`).

* Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

* Your Name - [Your Email](mailto:your.email@example.com)

* Project Link: [https://github.com/your_username/your_repository_name](https://github.com/your_username/your_repository_name)
