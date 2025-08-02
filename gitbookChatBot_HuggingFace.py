import os
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceTextGenInference # For Inference Endpoints
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document

# --- Constants ---
GITBOOK_DOCS_URL = "https://gitbook.com/docs" # Start at the base documentation URL

# Using a smaller, efficient model for embeddings
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Using a smaller language model suitable for Q&A on HuggingFace Hub
# Ensure you have access to this model on HuggingFace Hub with your token

# HuggingFace Inference Endpoint URL (now using a public endpoint)
HF_ENDPOINT_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

DB_PATH = "vectorstore_hf"
MAX_CRAWL_DEPTH = 2 # Limit the crawling depth to avoid going too wide
# Use a global set for visited URLs to manage state across recursive calls
VISITED_URLS = set()
# Set a user-agent to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# --- Helper Functions ---

def get_page_html(url: str, max_retries: int = 3, delay: int = 1) -> str:
    """Fetches the raw HTML content of a given URL with retries and delay."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url} (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to fetch {url} after {max_retries} attempts.")
                return ""

def extract_main_content(html_content: str, url: str) -> Dict[str, str]:
    """Extracts main text content from HTML using BeautifulSoup."""
    if not html_content:
        return {"content": "", "source_url": url}

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted elements that are typically not part of the main content
    unwanted_selectors = [
        'script', 'style', 'header', 'footer', 'nav',
        '.sidebar', '.navbar', '.page-footer', '.top-bar',
        'form', 'button', 'input', 'select', 'textarea', # Form elements
        '[id*="ads"]', '[class*="ad-"]', # Common ad selectors
        '.breadcrumb', '.pagination', '.toc', # Navigation/UI elements
        'svg', 'img', # Images and SVGs might not add text value
    ]
    for selector in unwanted_selectors:
        for elem in soup.select(selector):
            elem.decompose()

    # Priority order for finding the main content div/element
    # These selectors are common for main content areas. You might need to
    # inspect GitBook's specific HTML structure to refine them.
    main_content_div = (
        soup.find("div", class_="page-content") or # Common for GitBook
        soup.find("div", class_="main-content") or
        soup.find("article") or
        soup.find("main") or
        soup.find("div", {"role": "main"}) or # WAI-ARIA role
        soup.body
    )
    
    if main_content_div:
        # Get text, strip whitespace, and normalize multiple newlines to single ones
        text = main_content_div.get_text(separator='\n', strip=True)
        # Further clean by removing excessive blank lines
        text = '\n'.join(line for line in text.splitlines() if line.strip())
        return {"content": text, "source_url": url}
    
    return {"content": "", "source_url": url}

def find_internal_links(soup: BeautifulSoup, base_url: str, target_domain: str) -> List[str]:
    """Finds unique internal links within the specified target domain."""
    links = set() # Use a set to automatically handle uniqueness
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href) # Resolve relative URLs
        parsed_full_url = urlparse(full_url)
        
        # Ensure the link is within the target domain and not a fragment
        if (parsed_full_url.netloc == target_domain and 
            parsed_full_url.fragment == '' and
            not parsed_full_url.query # Ignore query parameters for simpler crawling
        ):
            # Normalize URL path to remove trailing slashes if not home page
            normalized_path = parsed_full_url.path
            if normalized_path != '/' and normalized_path.endswith('/'):
                normalized_path = normalized_path.rstrip('/')
            
            normalized_url = parsed_full_url._replace(path=normalized_path, query='', fragment='').geturl()
            links.add(normalized_url)
            
    return list(links)

def fetch_gitbook_content_recursive(url: str, current_depth: int = 0) -> List[Document]:
    """
    Fetch and parse content from GitBook documentation recursively.
    """
    global VISITED_URLS # Access the global set

    # Extract the base domain from the initial URL
    initial_parsed_url = urlparse(GITBOOK_DOCS_URL)
    target_domain = initial_parsed_url.netloc

    if url in VISITED_URLS or current_depth > MAX_CRAWL_DEPTH:
        return []

    VISITED_URLS.add(url)
    documents = []
    
    print(f"[{current_depth}/{MAX_CRAWL_DEPTH}] Crawling: {url}")
    
    html_content = get_page_html(url)
    if not html_content:
        return [] # Don't proceed if HTML content couldn't be fetched

    page_data = extract_main_content(html_content, url)
    if page_data["content"]:
        documents.append(
            Document(
                page_content=page_data["content"],
                metadata={
                    "source": page_data["source_url"],
                    "depth": current_depth
                }
            )
        )
    
    # Parse the fetched HTML to find internal links
    soup_for_links = BeautifulSoup(html_content, 'html.parser')
    internal_links = find_internal_links(soup_for_links, url, target_domain)
    
    for link in internal_links:
        # Only crawl if the link hasn't been visited and is within depth limit
        if link not in VISITED_URLS and current_depth + 1 <= MAX_CRAWL_DEPTH:
            documents.extend(fetch_gitbook_content_recursive(link, current_depth + 1))
    
    return documents

def create_vectorstore(documents: List[Document]):
    """Create and populate the vector store with document chunks."""
    if not documents:
        raise ValueError("No documents provided for vector store creation.")

    print("Initializing vector store...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Reverted to 500, 400 was very small for some contexts
        chunk_overlap=100, # Increased overlap for better context
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    print("Creating vector store (this may take a while for large datasets)...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Vector store created and persisted.")
    return vectorstore

def create_rag_chain(vectorstore):
    """Create the RAG chain combining retrieval and generation."""
    # For public endpoints, do not use or check for HuggingFace API token
    try:
        llm = HuggingFaceTextGenInference(
            inference_server_url=HF_ENDPOINT_URL,
            max_new_tokens=512,
            temperature=0.1
        )
    except Exception as e:
        print(f"Error initializing HuggingFaceTextGenInference LLM (endpoint: {HF_ENDPOINT_URL}): {e}")
        print("Please ensure the endpoint exists and is publicly accessible.")
        raise

    retriever = vectorstore.as_retriever(
        search_type="mmr", # Maximum Marginal Relevance for better diversity
        search_kwargs={
            "k": 4, # Increased k to retrieve slightly more relevant documents
            "fetch_k": 8, # Fetch more candidates for MMR to select from
            "lambda_mult": 0.7 # Balance between similarity and diversity (0.0 for diversity, 1.0 for similarity)
        }
    )

    template = """You are a helpful assistant that answers questions about GitBook documentation.
    Answer the question based on the context provided. If you don't know the answer, state that you don't know based on the provided context. Avoid making up answers.
    
    Context: {context}
    
    Question: {question}
    
    Answer:""" # Removed "Provide a clear and concise answer:" as it can sometimes make the LLM repeat it

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain

def main():
    # No HuggingFace API token needed for public endpoints

    os.makedirs(DB_PATH, exist_ok=True)
    
    try:
        vectorstore = None
        # Check if vector store exists and is not empty
        if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
            print("No existing vector store found or it's empty. Initializing new one...")
            # Reset visited URLs for a fresh crawl each time vector store is built
            global VISITED_URLS
            VISITED_URLS = set()
            documents = fetch_gitbook_content_recursive(GITBOOK_DOCS_URL)
            
            if not documents:
                raise ValueError("No content fetched from GitBook documentation. Cannot create vector store.")
            
            vectorstore = create_vectorstore(documents)
        else:
            print("Loading existing vector store...")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            print("Vector store loaded successfully.")

        chain = create_rag_chain(vectorstore)
        
        print("\nGitBook Assistant initialized! Type 'quit' to exit.")
        print("You can ask questions about GitBook's features, usage, and documentation.")
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("Thank you for using GitBook Assistant. Goodbye!")
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
                
            try:
                response = chain.invoke(question)
                print("\nAnswer:", response.strip())
            except Exception as e:
                print(f"Error generating response: {type(e).__name__}: {str(e)}")
                print("Please ensure the HuggingFace endpoint is correct and publicly accessible.")

    except ValueError as ve:
        print(f"Configuration Error: {str(ve)}")
    except Exception as e:
        print(f"An unhandled error occurred in main: {type(e).__name__}: {str(e)}")
        print("Please review the error message and ensure all dependencies are installed and configured.")

if __name__ == "__main__":
    main()