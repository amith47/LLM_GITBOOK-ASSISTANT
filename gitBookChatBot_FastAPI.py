import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# FastAPI for REST API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatbotConfig:
    """Configuration for the RAG chatbot"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama2"  # Requires Ollama to be running locally
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 4
    memory_window: int = 10
    vector_db_path: str = "./chroma_db"
    gitbook_base_url: str = "https://docs.gitbook.com"

class GitBookScraper:
    """Scraper for GitBook documentation"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_page_content(self, url: str) -> Optional[str]:
        """Extract text content from a GitBook page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content (GitBook specific selectors)
            content_selectors = [
                '[data-testid="page-content"]',
                '.markdown-body',
                'main',
                'article',
                '.content'
            ]
            
            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(strip=True, separator='\n')
                    break
            
            if not content:
                # Fallback to body content
                content = soup.get_text(strip=True, separator='\n')
            
            return content
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def discover_pages(self, max_pages: int = 50) -> List[str]:
        """Discover GitBook pages to scrape"""
        pages = []
        try:
            # Start with the main documentation page
            response = self.session.get(self.base_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all internal links
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                if href.startswith('/'):
                    full_url = urljoin(self.base_url, href)
                elif href.startswith(self.base_url):
                    full_url = href
                else:
                    continue
                
                # Filter for documentation pages
                if self._is_doc_page(full_url) and full_url not in pages:
                    pages.append(full_url)
                    if len(pages) >= max_pages:
                        break
            
            # If no pages found, add the base URL
            if not pages:
                pages.append(self.base_url)
                
        except Exception as e:
            logger.error(f"Error discovering pages: {e}")
            pages = [self.base_url]  # Fallback to base URL
        
        return pages[:max_pages]
    
    def _is_doc_page(self, url: str) -> bool:
        """Check if URL is likely a documentation page"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Skip certain paths
        skip_patterns = [
            '/api/', '/login', '/signup', '/pricing', 
            '/support', '/contact', '/about', '/blog',
            '.pdf', '.zip', '.jpg', '.png', '.gif'
        ]
        
        return not any(pattern in path for pattern in skip_patterns)

class RAGChatbot:
    """RAG-based chatbot using LangChain and ChromaDB"""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.conversation_chain = None
        
    async def initialize(self):
        """Initialize the chatbot components"""
        logger.info("Initializing RAG chatbot...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM (Ollama)
        try:
            self.llm = Ollama(
                model=self.config.llm_model,
                temperature=0.7,
                top_p=0.9
            )
            # Test the connection
            test_response = self.llm("Hello")
            logger.info("LLM connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Please ensure Ollama is running and the model is available")
            raise
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.config.vector_db_path)
        
        # Initialize vector store
        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=self.embeddings,
            collection_name="gitbook_docs"
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=self.config.memory_window,
            memory_key="chat_history",
            return_messages=True
        )
        
        logger.info("RAG chatbot initialized successfully")
    
    async def load_knowledge_base(self):
        """Load GitBook documentation into the vector database"""
        logger.info("Loading GitBook documentation...")
        
        scraper = GitBookScraper(self.config.gitbook_base_url)
        
        # Discover pages to scrape
        pages = scraper.discover_pages(max_pages=20)  # Limit for demo
        logger.info(f"Found {len(pages)} pages to process")
        
        documents = []
        
        for url in pages:
            logger.info(f"Processing: {url}")
            content = scraper.get_page_content(url)
            
            if content and len(content.strip()) > 100:  # Skip very short content
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "title": self._extract_title_from_url(url)
                    }
                )
                documents.append(doc)
        
        if not documents:
            logger.warning("No documents loaded. Using sample content.")
            # Add sample GitBook content for demonstration
            sample_docs = self._get_sample_gitbook_docs()
            documents.extend(sample_docs)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Add to vector store
        if split_docs:
            self.vectorstore.add_documents(split_docs)
            logger.info("Documents added to vector store successfully")
        
        # Setup retrieval chains
        self._setup_chains()
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract title from URL"""
        return url.split('/')[-1].replace('-', ' ').title() or "GitBook Documentation"
    
    def _get_sample_gitbook_docs(self) -> List[Document]:
        """Sample GitBook documentation for demonstration"""
        return [
            Document(
                page_content="""
                GitBook is a modern documentation platform that makes it easy to create, 
                organize and share knowledge. It provides a clean, intuitive interface 
                for writing and organizing documentation, with features like:
                
                - Rich text editing with markdown support
                - Collaborative editing and comments
                - Integration with Git repositories
                - Custom domains and branding
                - Analytics and insights
                - API documentation tools
                """,
                metadata={"source": "https://docs.gitbook.com/overview", "title": "GitBook Overview"}
            ),
            Document(
                page_content="""
                Getting started with GitBook is simple:
                
                1. Create your account at gitbook.com
                2. Create your first space
                3. Start writing your documentation
                4. Invite team members to collaborate
                5. Publish and share your documentation
                
                GitBook supports various content types including text, images, 
                code blocks, tables, and embeds. You can organize content using 
                pages and collections for better structure.
                """,
                metadata={"source": "https://docs.gitbook.com/getting-started", "title": "Getting Started"}
            ),
            Document(
                page_content="""
                GitBook API allows developers to programmatically interact with 
                GitBook content. Key features include:
                
                - Content management (create, read, update, delete)
                - Space and organization management
                - User and permission management
                - Webhook support for real-time updates
                - Integration with external systems
                
                The API uses REST architecture with JSON responses and supports 
                authentication via API tokens or OAuth.
                """,
                metadata={"source": "https://docs.gitbook.com/api", "title": "API Documentation"}
            )
        ]
    
    def _setup_chains(self):
        """Setup retrieval and conversation chains"""
        # Custom prompt template
        prompt_template = """
        You are a helpful assistant that answers questions about GitBook documentation.
        Use the following pieces of context to answer the user's question. If you don't 
        know the answer based on the context, just say you don't know - don't make up answers.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        
        Assistant: Let me help you with that GitBook question based on the documentation.
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Setup retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.max_retrieval_docs}
            ),
            return_source_documents=True
        )
        
        # Setup conversational retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.max_retrieval_docs}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
    
    async def get_response(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """Get response from the chatbot"""
        try:
            if use_memory and self.conversation_chain:
                result = self.conversation_chain({"question": question})
            else:
                result = self.qa_chain({"query": question})
            
            # Format response
            response = {
                "answer": result["answer"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": []
            }

# FastAPI Application
app = FastAPI(title="GitBook RAG Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    use_memory: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# Global chatbot instance
chatbot: Optional[RAGChatbot] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup"""
    global chatbot
    config = ChatbotConfig()
    chatbot = RAGChatbot(config)
    
    try:
        await chatbot.initialize()
        await chatbot.load_knowledge_base()
        logger.info("Chatbot startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "GitBook RAG Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the GitBook documentation bot"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    response = await chatbot.get_response(
        question=request.question,
        use_memory=request.use_memory
    )
    
    return ChatResponse(**response)

@app.post("/reset-memory")
async def reset_memory():
    """Reset conversation memory"""
    if chatbot and chatbot.memory:
        chatbot.memory.clear()
        return {"message": "Memory reset successfully"}
    return {"message": "No memory to reset"}

@app.get("/health")
async def health_check():
    """Health check with system status"""
    status = {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "vectorstore_ready": chatbot.vectorstore is not None if chatbot else False,
        "llm_ready": chatbot.llm is not None if chatbot else False
    }
    return status

if __name__ == "__main__":
    # Run the application
    print("Starting GitBook RAG Chatbot Backend...")
    print("Make sure Ollama is running with llama2 model installed!")
    print("Install Ollama: https://ollama.ai/")
    print("Install model: ollama pull llama2")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )