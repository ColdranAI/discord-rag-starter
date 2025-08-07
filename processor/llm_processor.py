"""
Production-ready LLM-enhanced RAG processor for Discord bot.
Handles document processing, embedding, retrieval, and LLM generation.
"""

import asyncio
import asyncpg
import nats
import json
import boto3
import httpx
import os
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import redis
from dataclasses import dataclass
import tiktoken

# Document processing
import PyPDF2
from io import BytesIO
import mimetypes

# Try to import python-magic, fallback to mimetypes if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available, using mimetypes fallback")

# ML/AI imports
from sentence_transformers import SentenceTransformer
import openai
import anthropic
import google.generativeai as genai
from exa_py import Exa

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingJob:
    """Structure for processing jobs"""
    job_type: str  # 'upload', 'query'
    user_id: str
    guild_id: Optional[str]
    data: Dict[str, Any]
    llm_provider: Optional[str] = None

@dataclass 
class RetrievalResult:
    """Structure for retrieval results"""
    content: str
    filename: str
    score: float
    metadata: Dict[str, Any]

class LLMProcessor:
    """Production-grade LLM processor for RAG operations"""
    
    def __init__(self):
        # Service connections
        self.db_pool = None
        self.nc = None
        self.s3 = None
        self.redis_client = None
        
        # ML models
        self.embedding_model = None
        self.tokenizer = None
        
        # LLM clients
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        self.exa_client = None
        
        # Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
        self.max_chunks_per_doc = int(os.getenv("MAX_CHUNKS_PER_DOC", "100"))
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "5"))
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
        
    async def initialize(self):
        """Initialize all connections and models"""
        logger.info("Initializing LLM Processor...")
        
        # Initialize database
        await self._init_database()
        
        # Initialize services
        await self._init_services()
        
        # Initialize ML models
        await self._init_models()
        
        # Initialize LLM clients
        await self._init_llm_clients()
        
        logger.info("LLM Processor initialized successfully")
    
    async def _init_database(self):
        """Initialize PostgreSQL connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=os.getenv("DB_HOST"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                min_size=2,
                max_size=10
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _init_services(self):
        """Initialize external services"""
        # NATS
        self.nc = await nats.connect(
            os.getenv("NATS_URL"),
            user=os.getenv("NATS_USER"),
            password=os.getenv("NATS_PASSWORD")
        )
        
        # S3
        self.s3 = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        # Redis
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        
        logger.info("External services initialized")
    
    async def _init_models(self):
        """Initialize ML models"""
        # Embedding model
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Tokenizer for counting tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        logger.info(f"ML models loaded: {model_name}")
    
    async def _init_llm_clients(self):
        """Initialize LLM clients based on configuration"""
        if os.getenv("USE_OPENAI", "false").lower() == "true":
            self.openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("OpenAI client initialized")
        
        if os.getenv("USE_ANTHROPIC", "false").lower() == "true":
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            logger.info("Anthropic client initialized")
        
        if os.getenv("USE_GEMINI", "false").lower() == "true":
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_client = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini client initialized")
        
        if os.getenv("USE_EXA", "false").lower() == "true":
            self.exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))
            logger.info("Exa client initialized")
    
    def detect_file_type(self, file_content: bytes, filename: str = "") -> str:
        """Detect file type using python-magic or fallback to mimetypes"""
        try:
            if MAGIC_AVAILABLE:
                mime_type = magic.from_buffer(file_content, mime=True)
                return mime_type
            else:
                # Fallback to mimetypes based on filename
                mime_type, _ = mimetypes.guess_type(filename)
                return mime_type or 'application/octet-stream'
        except Exception as e:
            logger.error(f"File type detection failed: {e}")
            # Final fallback based on filename extension
            if filename.lower().endswith('.pdf'):
                return 'application/pdf'
            elif filename.lower().endswith(('.txt', '.md')):
                return 'text/plain'
            else:
                return 'application/octet-stream'

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def detect_file_type(self, content: bytes) -> str:
        """Detect file type from content"""
        try:
            mime_type = magic.from_buffer(content, mime=True)
            return mime_type
        except:
            return "application/octet-stream"
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Intelligent text chunking with overlap and metadata preservation.
        
        Returns list of chunks with format:
        {
            'content': str,
            'metadata': dict,
            'chunk_index': int,
            'token_count': int
        }
        """
        if not text.strip():
            return []
        
        # Split by sentences/paragraphs for better semantic chunks
        sentences = text.replace('\n\n', ' [PARAGRAPH] ').split('. ')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            token_count = len(self.tokenizer.encode(potential_chunk))
            
            if token_count > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_tokens = len(self.tokenizer.encode(current_chunk))
                chunks.append({
                    'content': current_chunk.replace(' [PARAGRAPH] ', '\n\n'),
                    'metadata': {
                        **metadata,
                        'chunk_index': chunk_index,
                        'token_count': chunk_tokens,
                        'chunk_type': 'semantic'
                    },
                    'chunk_index': chunk_index,
                    'token_count': chunk_tokens
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split('. ')[-2:]  # Last 2 sentences
                current_chunk = '. '.join(overlap_sentences) + ". " + sentence if len(overlap_sentences) > 1 else sentence
                chunk_index += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunk_tokens = len(self.tokenizer.encode(current_chunk))
            chunks.append({
                'content': current_chunk.replace(' [PARAGRAPH] ', '\n\n'),
                'metadata': {
                    **metadata,
                    'chunk_index': chunk_index,
                    'token_count': chunk_tokens,
                    'chunk_type': 'semantic'
                },
                'chunk_index': chunk_index,
                'token_count': chunk_tokens
            })
        
        # Limit chunks per document
        if len(chunks) > self.max_chunks_per_doc:
            logger.warning(f"Document has {len(chunks)} chunks, limiting to {self.max_chunks_per_doc}")
            chunks = chunks[:self.max_chunks_per_doc]
        
        return chunks
    
    async def store_embeddings(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Store embeddings in Qdrant"""
        try:
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode(chunk['content']).tolist()
                
                # Create point for Qdrant
                point = {
                    "id": f"{doc_id}_{i}",
                    "vector": embedding,
                    "payload": {
                        "content": chunk['content'],
                        "document_id": doc_id,
                        "chunk_index": chunk['chunk_index'],
                        "token_count": chunk['token_count'],
                        **chunk['metadata']
                    }
                }
                points.append(point)
            
            # Store in Qdrant
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{os.getenv('QDRANT_URL')}/collections/documents/points",
                    headers={"api-key": os.getenv("QDRANT_API_KEY")},
                    json={"points": points},
                    timeout=30.0
                )
                
                if response.status_code not in [200, 201]:
                    logger.error(f"Qdrant storage failed: {response.status_code} - {response.text}")
                    return False
            
            logger.info(f"Stored {len(points)} embeddings for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Embedding storage failed: {e}")
            return False
    
    async def retrieve_relevant_chunks(self, query: str, user_id: str, k: int = None) -> List[RetrievalResult]:
        """Retrieve relevant chunks from Qdrant"""
        k = k or self.retrieval_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Qdrant
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{os.getenv('QDRANT_URL')}/collections/documents/points/search",
                    headers={"api-key": os.getenv("QDRANT_API_KEY")},
                    json={
                        "vector": query_embedding,
                        "limit": k * 2,  # Get more for filtering
                        "with_payload": True,
                        "filter": {
                            "must": [
                                {"key": "uploaded_by", "match": {"value": user_id}}
                            ]
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Qdrant search failed: {response.status_code}")
                    return []
                
                results = response.json().get("result", [])
                
                # Convert to RetrievalResult objects
                retrieval_results = []
                seen_docs = set()
                
                for result in results:
                    payload = result.get("payload", {})
                    doc_id = payload.get("document_id")
                    
                    # Limit results per document for diversity
                    if len([r for r in retrieval_results if r.metadata.get("document_id") == doc_id]) >= 2:
                        continue
                    
                    retrieval_results.append(RetrievalResult(
                        content=payload.get("content", ""),
                        filename=payload.get("filename", "Unknown"),
                        score=result.get("score", 0.0),
                        metadata=payload
                    ))
                    
                    if len(retrieval_results) >= k:
                        break
                
                return retrieval_results
                
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def load_prompt_template(self, template_name: str) -> str:
        """Load prompt template from external file"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompt_path = os.path.join(script_dir, "prompts", f"{template_name}.txt")
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Prompt template {template_name} not found, using default")
            return self._get_default_prompt_template(template_name)
        except Exception as e:
            logger.error(f"Error loading prompt template {template_name}: {e}")
            return self._get_default_prompt_template(template_name)
    
    def _get_default_prompt_template(self, template_name: str) -> str:
        """Fallback prompt templates"""
        if template_name == "rag_query_prompt":
            return """You are a helpful AI assistant. Answer the user's question using the provided context.

Document Context:
{document_context}

Web Search Results:
{web_context}

Question: {query}

Please provide a comprehensive answer citing your sources."""
        else:
            return "You are a helpful AI assistant. Please answer the user's question to the best of your ability."

    async def search_web_with_exa(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """Search the web using Exa AI"""
        if not self.exa_client:
            logger.warning("Exa client not initialized")
            return []
        
        try:
            # Use Exa's search_and_contents for comprehensive results
            search_result = self.exa_client.search_and_contents(
                query=query,
                text=True,
                highlights=True,
                num_results=num_results,
                type="auto"  # Automatic search type selection
            )
            
            web_results = []
            for result in search_result.results:
                web_results.append({
                    "title": result.title,
                    "url": result.url,
                    "content": result.text[:1000] if result.text else "",  # Limit content length
                    "highlights": result.highlights[:3] if result.highlights else [],  # Top 3 highlights
                    "score": getattr(result, 'score', 1.0)
                })
            
            logger.info(f"Exa search returned {len(web_results)} results for query: {query}")
            return web_results
            
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return []

    def construct_enhanced_prompt(self, query: str, context_chunks: List[RetrievalResult], 
                                web_results: List[Dict[str, Any]] = None) -> str:
        """Construct enhanced prompt with both document context and web search results"""
        # Load the prompt template
        template = self.load_prompt_template("rag_query_prompt")
        
        # Build document context
        document_context = ""
        if context_chunks:
            doc_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                doc_parts.append(f"[Document {i}: {chunk.filename}]\n{chunk.content}\n")
            document_context = "\n".join(doc_parts)
        else:
            document_context = "No relevant documents found in personal library."
        
        # Build web context
        web_context = ""
        if web_results:
            web_parts = []
            for i, result in enumerate(web_results, 1):
                highlights_text = ""
                if result.get('highlights'):
                    highlights_text = f"\nKey points: {'; '.join(result['highlights'])}"
                
                web_parts.append(
                    f"[Web Source {i}: {result['title']}]\n"
                    f"URL: {result['url']}\n"
                    f"Content: {result['content']}{highlights_text}\n"
                )
            web_context = "\n".join(web_parts)
        else:
            web_context = "No web search results available."
        
        # Fill in the template
        prompt = template.format(
            document_context=document_context,
            web_context=web_context,
            query=query
        )
        
        # Check token count and truncate if necessary
        token_count = len(self.tokenizer.encode(prompt))
        max_tokens = 4000  # Leave room for response
        
        if token_count > max_tokens:
            logger.warning(f"Prompt too long ({token_count} tokens), truncating content")
            # Truncate web content first, then document content if needed
            if web_results:
                # Reduce web results
                for result in web_results:
                    if result['content']:
                        result['content'] = result['content'][:500]
                web_parts = []
                for i, result in enumerate(web_results, 1):
                    web_parts.append(
                        f"[Web Source {i}: {result['title']}]\n"
                        f"Content: {result['content']}\n"
                    )
                web_context = "\n".join(web_parts)
            
            # If still too long, truncate document content
            if len(self.tokenizer.encode(prompt)) > max_tokens and context_chunks:
                while len(self.tokenizer.encode(prompt)) > max_tokens and context_chunks:
                    context_chunks.pop()
                    doc_parts = []
                    for i, chunk in enumerate(context_chunks, 1):
                        doc_parts.append(f"[Document {i}: {chunk.filename}]\n{chunk.content}\n")
                    document_context = "\n".join(doc_parts) if doc_parts else "Document context truncated due to length."
            
            # Rebuild prompt with truncated content
            prompt = template.format(
                document_context=document_context,
                web_context=web_context,
                query=query
            )
        
        return prompt

    def construct_llm_prompt(self, query: str, context_chunks: List[RetrievalResult]) -> str:
        """Legacy method - now redirects to enhanced prompt construction"""
        return self.construct_enhanced_prompt(query, context_chunks, None)
    
    async def call_llm(self, prompt: str, llm_provider: str) -> Optional[str]:
        """Call specified LLM provider with proper error handling and retries"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                if llm_provider == "openai" and self.openai_client:
                    response = await self._call_openai(prompt)
                elif llm_provider == "anthropic" and self.anthropic_client:
                    response = await self._call_anthropic(prompt)
                elif llm_provider == "gemini" and self.gemini_client:
                    response = await self._call_gemini(prompt)
                else:
                    logger.error(f"Unsupported or unconfigured LLM provider: {llm_provider}")
                    return None
                
                if response:
                    return response
                    
            except Exception as e:
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All LLM attempts failed for {llm_provider}")
        
        return None
    
    async def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        try:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API"""
        try:
            response = self.anthropic_client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini API"""
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.1
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def get_cache_key(self, query: str, user_id: str, llm_provider: str) -> str:
        """Generate cache key for query results"""
        content = f"{query}:{user_id}:{llm_provider}"
        return f"rag:query:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def cache_response(self, cache_key: str, response: str):
        """Cache response for future use"""
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, response)
            logger.info(f"Cached response for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def log_interaction(self, user_id: str, query: str, llm_provider: str, 
                            response: str, processing_time: float, token_count: int):
        """Log interaction for monitoring and debugging"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO llm_interactions 
                    (user_id, query, llm_provider, response, processing_time, token_count, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, user_id, query[:500], llm_provider, response[:1000], 
                processing_time, token_count, datetime.utcnow())
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
    
    async def process_document_upload(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process document upload job"""
        start_time = time.time()
        doc_data = job.data
        doc_id = doc_data['id']
        filename = doc_data['filename']
        user_id = job.user_id
        
        try:
            logger.info(f"Processing document upload: {doc_id} ({filename})")
            
            # Download file from S3
            s3_key = doc_data.get('s3_key', f"documents/{user_id}/{doc_id}")
            obj = self.s3.get_object(Bucket=os.getenv("AWS_BUCKET"), Key=s3_key)
            content = obj['Body'].read()
            
            # Validate file size
            if len(content) > self.max_file_size:
                raise ValueError(f"File too large: {len(content)} bytes")
            
            # Extract text based on file type
            file_type = self.detect_file_type(content)
            
            if file_type == "application/pdf":
                text = self.extract_text_from_pdf(content)
            elif file_type.startswith("text/") or filename.lower().endswith(('.txt', '.md', '.mdx', '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.xml', '.yaml', '.yml', '.log', '.csv', '.toml', '.ini', '.conf')):
                text = content.decode('utf-8', errors='ignore')
            elif filename.lower().endswith('.json'):
                # Handle JSON files
                try:
                    import json
                    json_data = json.loads(content.decode('utf-8', errors='ignore'))
                    text = json.dumps(json_data, indent=2)  # Pretty print for better chunking
                except Exception:
                    text = content.decode('utf-8', errors='ignore')  # Fallback to raw text
            else:
                # Try as text anyway
                text = content.decode('utf-8', errors='ignore')
            
            if not text.strip():
                raise ValueError("No text content extracted from file")
            
            # Prepare metadata
            metadata = {
                'filename': filename,
                'uploaded_by': user_id,
                'file_type': file_type,
                'file_size': len(content),
                'upload_date': datetime.utcnow().isoformat(),
                'document_id': doc_id
            }
            
            # Chunk text
            chunks = self.chunk_text(text, metadata)
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            # Store embeddings
            success = await self.store_embeddings(doc_id, chunks)
            
            if not success:
                raise Exception("Failed to store embeddings")
            
            # Update database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE documents 
                    SET processed = true, text_preview = $1, chunk_count = $2
                    WHERE id = $3
                """, text[:500], len(chunks), doc_id)
            
            processing_time = time.time() - start_time
            logger.info(f"Document {doc_id} processed successfully in {processing_time:.2f}s")
            
            return {
                'status': 'success',
                'doc_id': doc_id,
                'chunks_created': len(chunks),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Document processing failed for {doc_id}: {e}")
            
            # Update database with error
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE documents 
                        SET processed = false, error_message = $1
                        WHERE id = $2
                    """, str(e), doc_id)
            except:
                pass
            
            return {
                'status': 'error',
                'doc_id': doc_id,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def process_query(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process query job with LLM enhancement and optional web search"""
        start_time = time.time()
        query = job.data['query']
        llm_provider = job.llm_provider
        user_id = job.user_id
        use_web_search = job.data.get('use_web_search', True)  # Default to True for enhanced answers
        
        try:
            logger.info(f"Processing query for user {user_id} with {llm_provider}: {query[:100]}")
            
            # Check cache first
            cache_key = self.get_cache_key(query, user_id, f"{llm_provider}_web_{use_web_search}")
            cached_response = await self.get_cached_response(cache_key)
            
            if cached_response:
                return {
                    'status': 'success',
                    'response': cached_response,
                    'cached': True,
                    'processing_time': time.time() - start_time
                }
            
            # Retrieve relevant chunks from documents
            relevant_chunks = await self.retrieve_relevant_chunks(query, user_id)
            
            # Search web if enabled and Exa is available
            web_results = []
            if use_web_search and self.exa_client:
                try:
                    web_results = await self.search_web_with_exa(query, num_results=3)
                except Exception as e:
                    logger.warning(f"Web search failed, continuing without: {e}")
            
            # If no document results and no web results, return helpful message
            if not relevant_chunks and not web_results:
                response = "I couldn't find any relevant information to answer your question. Try uploading some documents first, rephrasing your question, or check your internet connection for web search."
                return {
                    'status': 'success',
                    'response': response,
                    'sources': [],
                    'web_sources': [],
                    'processing_time': time.time() - start_time
                }
            
            # Construct enhanced prompt with both document and web context
            prompt = self.construct_enhanced_prompt(query, relevant_chunks, web_results)
            
            # Call LLM
            llm_response = await self.call_llm(prompt, llm_provider)
            
            if not llm_response:
                # Fallback to summarized content
                response_parts = []
                if relevant_chunks:
                    response_parts.append(f"Found {len(relevant_chunks)} relevant documents:")
                    for i, chunk in enumerate(relevant_chunks[:3], 1):
                        response_parts.append(f"**{chunk.filename}**: {chunk.content[:200]}...")
                
                if web_results:
                    response_parts.append(f"\nWeb search results:")
                    for i, result in enumerate(web_results[:2], 1):
                        response_parts.append(f"**{result['title']}**: {result['content'][:200]}...")
                
                response = "\n\n".join(response_parts) if response_parts else "Unable to generate response."
            else:
                response = llm_response
                # Cache successful response
                await self.cache_response(cache_key, response)
            
            # Log interaction
            token_count = len(self.tokenizer.encode(prompt + (llm_response or "")))
            await self.log_interaction(
                user_id, query, llm_provider, response, 
                time.time() - start_time, token_count
            )
            
            # Prepare sources
            doc_sources = [
                {
                    'type': 'document',
                    'filename': chunk.filename,
                    'score': chunk.score,
                    'preview': chunk.content[:150] + "..."
                }
                for chunk in relevant_chunks[:3]
            ]
            
            web_sources = [
                {
                    'type': 'web',
                    'title': result['title'],
                    'url': result['url'],
                    'preview': result['content'][:150] + "..."
                }
                for result in web_results[:3]
            ]
            
            return {
                'status': 'success',
                'response': response,
                'sources': doc_sources,
                'web_sources': web_sources,
                'cached': False,
                'processing_time': time.time() - start_time,
                'used_web_search': bool(web_results)
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def handle_job(self, msg):
        """Main job handler for NATS messages"""
        try:
            # Parse job
            job_data = json.loads(msg.data.decode())
            job = ProcessingJob(
                job_type=job_data['job_type'],
                user_id=job_data['user_id'],
                guild_id=job_data.get('guild_id'),
                data=job_data['data'],
                llm_provider=job_data.get('llm_provider')
            )
            
            # Process based on job type
            if job.job_type == 'upload':
                result = await self.process_document_upload(job)
            elif job.job_type == 'query':
                result = await self.process_query(job)
            else:
                result = {'status': 'error', 'error': f'Unknown job type: {job.job_type}'}
            
            # Send result back
            response_subject = job_data.get('response_subject')
            if response_subject:
                await self.nc.publish(response_subject, json.dumps(result).encode())
            
        except Exception as e:
            logger.error(f"Job handling failed: {e}")
            # Send error response if possible
            try:
                response_subject = json.loads(msg.data.decode()).get('response_subject')
                if response_subject:
                    error_result = {'status': 'error', 'error': str(e)}
                    await self.nc.publish(response_subject, json.dumps(error_result).encode())
            except:
                pass
    
    async def run(self):
        """Main worker loop"""
        await self.initialize()
        
        # Set up database schema
        await self._setup_database_schema()
        
        # Subscribe to job queues
        await self.nc.subscribe("jobs.upload", cb=self.handle_job)
        await self.nc.subscribe("jobs.query", cb=self.handle_job)
        
        logger.info("LLM Processor is running...")
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down LLM Processor...")
        finally:
            await self.cleanup()
    
    async def _setup_database_schema(self):
        """Set up additional database tables"""
        async with self.db_pool.acquire() as conn:
            # LLM interactions log
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_interactions (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    query TEXT NOT NULL,
                    llm_provider VARCHAR(50) NOT NULL,
                    response TEXT,
                    processing_time FLOAT,
                    token_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index separately (PostgreSQL syntax)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_interactions_user_timestamp 
                ON llm_interactions(user_id, timestamp)
            """)
            
            # Update documents table if needed
            await conn.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS error_message TEXT
            """)
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.nc:
            await self.nc.close()
        if self.db_pool:
            await self.db_pool.close()

async def main():
    """Main entry point"""
    processor = LLMProcessor()
    await processor.run()

if __name__ == "__main__":
    asyncio.run(main())
