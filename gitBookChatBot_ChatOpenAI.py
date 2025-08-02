import os
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document

# Constants
GITBOOK_DOCS_URL = "https://gitbook.com/docs/docs"
# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable using:\n"
                    "For Windows PowerShell: $env:OPENAI_API_KEY = 'your-key'\n"
                    "For Command Prompt: set OPENAI_API_KEY=your-key")
DB_PATH = "vectorstore"
MAX_CRAWL_DEPTH = 2 # Limit the crawling depth to avoid going too wide
VISITED_URLS = set() # To prevent infinite loops during crawling

def get_page_content(url: str) -> Dict[str, str]:
    """
    Fetches content from a single URL and extracts main text.
    Returns a dictionary with 'content' and 'source_url'.
    """
    try:
        response = requests.get(url, timeout=10) # Add a timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"content": "", "source_url": url}

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script, style, header, footer, and navigation elements
    for unwanted_tag in soup(["script", "style", "header", "footer", "nav", ".nav-link", ".sidebar"]):
        unwanted_tag.decompose()

    # Attempt to find the main content area (you might need to adjust these selectors)
    main_content_div = soup.find("div", class_="main-content") or \
                       soup.find("article") or \
                       soup.find("main") or \
                       soup.body

    if main_content_div:
        content = main_content_div.get_text(strip=True, separator='\n')
    else:
        content = soup.get_text(strip=True, separator='\n')
    
    return {"content": content, "source_url": url}

def find_internal_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Finds internal links within the same domain."""
    links = []
    base_netloc = urlparse(base_url).netloc
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        parsed_full_url = urlparse(full_url)
        
        # Check if it's an internal link and not a fragment
        if parsed_full_url.netloc == base_netloc and parsed_full_url.fragment == '':
            links.append(full_url)
    return list(set(links)) # Return unique links

def fetch_gitbook_content_recursive(base_url: str, current_depth: int = 0) -> List[Document]:
    """
    Fetch and parse content from GitBook documentation recursively,
    respecting MAX_CRAWL_DEPTH.
    """
    global VISITED_URLS # Declare VISITED_URLS as global

    if base_url in VISITED_URLS or current_depth > MAX_CRAWL_DEPTH:
        return []

    VISITED_URLS.add(base_url)
    documents = []
    
    print(f"Crawling: {base_url} (Depth: {current_depth})")
    page_data = get_page_content(base_url)
    
    if page_data["content"]:
        documents.append(
            Document(
                page_content=page_data["content"],
                metadata={"source": page_data["source_url"]}
            )
        )
    
    # Find and crawl internal links
    soup = BeautifulSoup(requests.get(base_url).text, 'html.parser') # Re-fetch just for links
    internal_links = find_internal_links(soup, base_url)
    
    for link in internal_links:
        documents.extend(fetch_gitbook_content_recursive(link, current_depth + 1))
            
    return documents

def create_vectorstore(documents: List[Document]):
    """Create and populate the vector store with document chunks."""
    if not documents:
        print("No documents to process for vector store creation. Exiting.")
        exit(1)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    # Split documents into chunks
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    # Initialize embeddings using OpenAI
    embeddings = OpenAIEmbeddings()

    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Vector store created and persisted.")
    return vectorstore

def create_rag_chain(vectorstore):
    """Create the RAG chain combining retrieval and generation."""
    # Initialize the LLM using OpenAI
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
    except Exception as e:
        print(f"Error initializing OpenAI chat model: {e}")
        print("Please check your OpenAI API key and internet connection.")
        exit(1)

    # Create the retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Create the prompt template
    template = """You are a helpful assistant that answers questions about GitBook documentation.
    Use only the following context to answer the question. If you don't know the answer, say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    # Create and return the RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain

def main():
    vectorstore = None
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH): # Check if directory is empty
        print("Vector store not found or empty. Fetching GitBook documentation and creating it...")
        # Reset visited URLs for fresh crawl
        global VISITED_URLS
        VISITED_URLS = set() 
        documents = fetch_gitbook_content_recursive(GITBOOK_DOCS_URL)
        if documents:
            vectorstore = create_vectorstore(documents)
        else:
            print("No content fetched. Cannot create vector store. Exiting.")
            exit(1)
    else:
        print("Loading existing vector store...")
        # Load existing vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        print("Vector store loaded.")

    # Create the RAG chain
    chain = create_rag_chain(vectorstore)
    
    print("ChatBot initialized! Type 'quit' to exit.")
    
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'quit':
            print("Exiting ChatBot. Goodbye!")
            break
            
        try:
            response = chain.invoke(question)
            print("\nAnswer:", response)
        except Exception as e:
            print(f"An error occurred while invoking the chain: {str(e)}")
            # Optional: log full traceback for debugging
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()