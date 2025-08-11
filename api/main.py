import os
import json
import uuid
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncpg
import httpx
import nats
import boto3
import redis
from dotenv import load_dotenv
import logging
import resource

# LLM imports
from openai import AsyncOpenAI
import google.generativeai as genai
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()

# Security Configuration
ADMIN_DISCORD_ID = os.getenv("ADMIN_DISCORD_ID")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
ALLOWED_GUILD_ID = os.getenv("ALLOWED_GUILD_ID")
# No IP whitelisting needed - private container network
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
ALLOWED_FILE_EXTENSIONS = os.getenv("ALLOWED_FILE_EXTENSIONS", ".pdf,.txt,.md,.mdx,.json,.csv,.log,.py,.js,.ts,.tsx,.jsx,.html,.css,.xml,.yaml,.yml,.toml,.ini,.conf").split(",")

# LLM Configuration
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")  # fallback model
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER")
available_llm_providers: List[str] = []

# Disable docs in production
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
docs_url = "/docs" if ENVIRONMENT == "development" else None
redoc_url = "/redoc" if ENVIRONMENT == "development" else None

app = FastAPI(
    title="Discord RAG Bot API",
    description="Secure API for Discord RAG Bot - Single User",
    version="1.0.0",
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url="/openapi.json" if ENVIRONMENT == "development" else None
)

# No additional middleware needed - running in private Zerops containers

# Service connections
s3 = None
nc = None
db_pool = None
redis_client = None

# LLM clients
openai_client = None
gemini_client = None
anthropic_client = None

# Security models
class DiscordRequest(BaseModel):
    user_id: str
    guild_id: Optional[str] = None
    content: Optional[str] = None

# Rate limiting storage
rate_limit_store = {}

def rate_limit(max_requests: int = 10, window_minutes: int = 1):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or args[0] if args else None
            if not request:
                return await func(*args, **kwargs)
            
            client_ip = request.client.host
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=window_minutes)
            
            # Clean old entries
            if client_ip in rate_limit_store:
                rate_limit_store[client_ip] = [
                    timestamp for timestamp in rate_limit_store[client_ip] 
                    if timestamp > window_start
                ]
            else:
                rate_limit_store[client_ip] = []
            
            # Check rate limit
            if len(rate_limit_store[client_ip]) >= max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Add current request
            rate_limit_store[client_ip].append(now)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def verify_api_secret(x_api_secret: str = Header(None)):
    """Verify API secret header"""
    if not x_api_secret or not API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="API secret required")
    
    if not hmac.compare_digest(x_api_secret, API_SECRET_KEY):
        raise HTTPException(status_code=401, detail="Invalid API secret")
    
    return True

def verify_discord_user(
    x_discord_user_id: str = Header(None),
    x_discord_guild_id: str = Header(None, alias="X-Discord-Guild-ID")
):
    """Verify Discord user and guild"""
    if not x_discord_user_id:
        raise HTTPException(status_code=401, detail="Discord user ID required")
    
    if x_discord_user_id != ADMIN_DISCORD_ID:
        raise HTTPException(status_code=403, detail="Unauthorized Discord user")
    
    if ALLOWED_GUILD_ID and x_discord_guild_id != ALLOWED_GUILD_ID:
        raise HTTPException(status_code=403, detail="Unauthorized Discord guild")
    
    return {"user_id": x_discord_user_id, "guild_id": x_discord_guild_id}

# No IP verification needed - private container network

def verify_file_security(file: UploadFile):
    """Verify file security constraints"""
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE} bytes")
    
    # Check file extension
    if file.filename:
        file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if file_ext not in ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"File extension not allowed. Allowed extensions: {ALLOWED_FILE_EXTENSIONS}")
    else:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Check filename for path traversal
    if file.filename and (".." in file.filename or "/" in file.filename or "\\" in file.filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    return True

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response

@app.on_event("startup")
async def startup():
    global nc, db_pool, s3, redis_client, openai_client, gemini_client, anthropic_client, DEFAULT_LLM_PROVIDER, available_llm_providers
    import asyncio
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if ENVIRONMENT == "development" else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Validate required security environment variables
    required_vars = ["ADMIN_DISCORD_ID", "API_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("Starting secure Discord RAG API...")
    
    # Log memory usage
    try:
        memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_mb = memory_kb / 1024 if memory_kb > 100000 else memory_kb / (1024 * 1024)
        logger.info(f"Startup memory usage: {memory_mb:.2f}MB")
    except Exception as e:
        logger.info(f"Could not get memory usage: {e}")

    # NATS connection with authentication and retry
    logger.info("Starting NATS connection...")
    for attempt in range(5):
        try:
            nc = await nats.connect(
                os.getenv("NATS_URL"),
                user=os.getenv("NATS_USER"),
                password=os.getenv("NATS_PASSWORD")
            )
            logger.info("NATS connection successful")
            break
        except Exception as e:
            logger.error(f"NATS connection attempt {attempt + 1} failed: {e}")
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)

    # PostgreSQL connection with retry
    logger.info("Starting PostgreSQL connection...")
    for attempt in range(10):
        try:
            db_pool = await asyncpg.create_pool(
                host=os.getenv("DB_HOST"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                min_size=1,
                max_size=3
            )
            logger.info("PostgreSQL connection successful")
            break
        except Exception as e:
            logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt == 9:
                raise
            await asyncio.sleep(2 ** min(attempt, 4))

    # S3 client
    logger.info("Initializing S3 client...")
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("AWS_ENDPOINT"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        use_ssl=True,
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    # Redis client with retry
    logger.info("Starting Redis connection...")
    for attempt in range(5):
        try:
            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST"),
                port=6379,
                decode_responses=True
            )
            redis_client.ping()
            logger.info("Redis connection successful")
            break
        except Exception as e:
            logger.error(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)

    # Initialize database schema
    logger.info("Initializing database schema...")
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                filename VARCHAR(255),
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                text_preview TEXT,
                uploaded_by VARCHAR(255),
                file_hash VARCHAR(64)
            )
        """)
        
        # Add audit log table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                action VARCHAR(100),
                resource_id VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address INET,
                details JSONB
            )
        """)

    # Initialize LLM clients
    logger.info("Initializing LLM clients...")

    if USE_OPENAI and OPENAI_API_KEY:
        try:
            openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            available_llm_providers.append("openai")
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    if USE_GEMINI and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_client = genai.GenerativeModel('gemini-pro')
            available_llm_providers.append("gemini")
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    if USE_ANTHROPIC and ANTHROPIC_API_KEY:
        try:
            anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            available_llm_providers.append("anthropic")
            logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")

    # Determine default LLM provider if not set
    if not DEFAULT_LLM_PROVIDER or DEFAULT_LLM_PROVIDER not in available_llm_providers:
        if available_llm_providers:
            DEFAULT_LLM_PROVIDER = available_llm_providers[0]
            logger.info(f"Default LLM provider set to {DEFAULT_LLM_PROVIDER}")
        else:
            logger.warning("No LLM providers configured")

async def log_audit(user_id: str, action: str, resource_id: str = None, request: Request = None, details: Dict[str, Any] = None):
    """Log audit events"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO audit_log (user_id, action, resource_id, ip_address, details)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, action, resource_id, request.client.host if request else None, json.dumps(details or {}))
    except Exception as e:
        logging.error(f"Failed to log audit event: {e}")

async def generate_llm_response(query: str, context_documents: List[Dict], llm_provider: str = None) -> str:
    """Generate LLM-enhanced response using the specified provider"""
    global openai_client, gemini_client, anthropic_client, DEFAULT_LLM_PROVIDER

    provider = llm_provider or DEFAULT_LLM_PROVIDER

    # Build context from search results
    context = ""
    if context_documents:
        context = "\n\n".join([
            f"Document: {doc.get('payload', {}).get('filename', 'Unknown')}\n"
            f"Content: {doc.get('payload', {}).get('text', '')[:500]}..."
            for doc in context_documents[:3]
        ])

    # Create prompt
    prompt = f"""Based on the following document excerpts, please answer the user's question. If the documents don't contain relevant information, say so.

Context from documents:
{context}

User question: {query}

Please provide a helpful and accurate response based on the available context."""

    try:
        if provider == "openai" and openai_client:
            response = await openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content

        elif provider == "gemini" and gemini_client:
            response = await gemini_client.generate_content_async(prompt)
            return response.text

        elif provider == "anthropic" and anthropic_client:
            response = await anthropic_client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        else:
            if context_documents:
                return f"Found {len(context_documents)} relevant documents. Here are the key excerpts:\n\n" + \
                       "\n\n".join([
                           f"**{doc.get('payload', {}).get('filename', 'Unknown')}**: {doc.get('payload', {}).get('text', '')[:200]}..."
                           for doc in context_documents[:2]
                       ])
            else:
                return "No relevant documents found for your query."

    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        if context_documents:
            return f"Found {len(context_documents)} relevant documents. Here are the key excerpts:\n\n" + \
                   "\n\n".join([
                       f"**{doc.get('payload', {}).get('filename', 'Unknown')}**: {doc.get('payload', {}).get('text', '')[:200]}..."
                       for doc in context_documents[:2]
                   ])
        else:
            return "No relevant documents found for your query."

@app.post("/upload")
@rate_limit(max_requests=5, window_minutes=1)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    _api_secret: bool = Depends(verify_api_secret),
    discord_user: dict = Depends(verify_discord_user),
    _file_check: bool = Depends(verify_file_security)
):
    logger = logging.getLogger(__name__)
    
    # Generate unique ID
    doc_id = str(uuid.uuid4())
    logger.info(f"Starting upload for file: {file.filename}, assigned ID: {doc_id}")
    
    # Read file content and calculate hash
    file_content = await file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    
    # Check for duplicate files
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT id FROM documents WHERE file_hash = $1", file_hash)
        if existing:
            await log_audit(discord_user["user_id"], "upload_duplicate", doc_id, request, {"filename": file.filename, "hash": file_hash})
            raise HTTPException(status_code=409, detail="File already exists")

    # Save to S3 with secure key
    s3_key = f"documents/{discord_user['user_id']}/{doc_id}.{file.filename.split('.')[-1] if '.' in file.filename else 'bin'}"
    s3.put_object(
        Bucket=os.getenv("AWS_BUCKET"),
        Key=s3_key,
        Body=file_content,
        ServerSideEncryption='AES256',
        Metadata={
            'uploaded_by': discord_user["user_id"],
            'original_filename': file.filename,
            'upload_timestamp': datetime.utcnow().isoformat()
        }
    )
    logger.info(f"File {doc_id} saved to S3 successfully")

    # Save metadata to PostgreSQL
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO documents (id, filename, uploaded_by, file_hash)
            VALUES ($1, $2, $3, $4)
        """, doc_id, file.filename, discord_user["user_id"], file_hash)
    logger.info(f"File {doc_id} metadata saved to PostgreSQL")

    # Queue for processing
    await nc.publish("document.process", json.dumps({
        "id": doc_id,
        "filename": file.filename,
        "uploaded_by": discord_user["user_id"],
        "s3_key": s3_key
    }).encode())
    logger.info(f"File {doc_id} queued for processing via NATS")
    
    # Log audit event
    await log_audit(discord_user["user_id"], "upload", doc_id, request, {"filename": file.filename})

    return {"id": doc_id, "status": "queued", "filename": file.filename}

@app.get("/search")
@rate_limit(max_requests=20, window_minutes=1)
async def search(
    query: str,
    request: Request,
    llm: Optional[str] = None,  # openai, gemini, anthropic, or None
    _api_secret: bool = Depends(verify_api_secret),
    discord_user: dict = Depends(verify_discord_user)
):
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
    
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long")

    # Determine and validate LLM provider
    llm = llm or DEFAULT_LLM_PROVIDER
    if llm and llm not in ["openai", "gemini", "anthropic"]:
        raise HTTPException(status_code=400, detail="Invalid LLM provider. Use: openai, gemini, or anthropic")
    
    # Check cache first (include LLM provider in cache key)
    cache_key = f"search:{hashlib.md5((query + (llm or 'none')).encode()).hexdigest()}"
    cached = redis_client.get(cache_key)

    if cached:
        await log_audit(discord_user["user_id"], "search_cached", None, request, {"query": query[:100], "llm": llm})
        return json.loads(cached)

    # Call Qdrant for vector search
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{os.getenv('QDRANT_URL')}/collections/documents/points/search",
            headers={
                "api-key": os.getenv("QDRANT_API_KEY")
            },
            json={
                "vector": [0.1] * 384,  # This should be replaced with actual embedding
                "limit": 5,
                "with_payload": True,
                "filter": {
                    "must": [
                        {"key": "uploaded_by", "match": {"value": discord_user["user_id"]}}
                    ]
                }
            }
        )

    search_results = response.json().get("result", [])
    
    # Generate LLM response if requested
    llm_response = None
    if llm and search_results:
        try:
            llm_response = await generate_llm_response(query, search_results, llm)
        except Exception as e:
            logging.error(f"LLM response generation failed: {e}")

    result = {
        "query": query,
        "results": search_results,
        "llm_provider": llm,
        "llm_response": llm_response
    }

    # Cache for 5 minutes
    redis_client.setex(cache_key, 300, json.dumps(result))
    
    # Log audit event
    await log_audit(discord_user["user_id"], "search", None, request, {"query": query[:100], "llm": llm, "results_count": len(search_results)})

    return result

@app.get("/documents")
@rate_limit(max_requests=10, window_minutes=1)
async def list_documents(
    request: Request,
    _api_secret: bool = Depends(verify_api_secret),
    discord_user: dict = Depends(verify_discord_user)
):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, filename, upload_date, processed
            FROM documents
            WHERE uploaded_by = $1
            ORDER BY upload_date DESC
            LIMIT 20
        """, discord_user["user_id"])

    await log_audit(discord_user["user_id"], "list_documents", None, request)
    return [dict(row) for row in rows]

@app.delete("/documents/{doc_id}")
@rate_limit(max_requests=5, window_minutes=1)
async def delete_document(
    doc_id: str,
    request: Request,
    _api_secret: bool = Depends(verify_api_secret),
    discord_user: dict = Depends(verify_discord_user)
):
    async with db_pool.acquire() as conn:
        # Verify ownership
        doc = await conn.fetchrow("SELECT * FROM documents WHERE id = $1 AND uploaded_by = $2", doc_id, discord_user["user_id"])
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from database
        await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
    
    # Delete from S3
    try:
        s3_key = f"documents/{discord_user['user_id']}/{doc_id}"
        s3.delete_object(Bucket=os.getenv("AWS_BUCKET"), Key=s3_key)
    except Exception as e:
        logging.error(f"Failed to delete from S3: {e}")
    
    # Delete from Qdrant
    try:
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{os.getenv('QDRANT_URL')}/collections/documents/points/{doc_id}",
                headers={"api-key": os.getenv("QDRANT_API_KEY")}
            )
    except Exception as e:
        logging.error(f"Failed to delete from Qdrant: {e}")
    
    await log_audit(discord_user["user_id"], "delete", doc_id, request)
    return {"status": "deleted", "id": doc_id}

@app.post("/submit_job")
@rate_limit(max_requests=10, window_minutes=1)
async def submit_job(
    request: Request,
    job_request: dict,
    _api_secret: bool = Depends(verify_api_secret),
    discord_user: dict = Depends(verify_discord_user)
):
    """Submit a job to the NATS queue for processing"""
    try:
        # Generate unique response subject
        response_subject = f"response.{uuid.uuid4()}"
        
        # Prepare job payload
        job_payload = {
            "job_type": job_request.get("job_type"),
            "user_id": discord_user["user_id"],
            "guild_id": discord_user.get("guild_id"),
            "data": job_request.get("data", {}),
            "llm_provider": job_request.get("llm_provider"),
            "response_subject": response_subject
        }
        
        # Submit job to appropriate queue
        job_type = job_request.get("job_type")
        if job_type == "query":
            await nc.publish("jobs.query", json.dumps(job_payload).encode())
        elif job_type == "upload":
            await nc.publish("jobs.upload", json.dumps(job_payload).encode())
        else:
            raise HTTPException(status_code=400, detail="Invalid job type")
        
        # Wait for response with timeout
        try:
            # Subscribe to response
            sub = await nc.subscribe(response_subject)
            msg = await sub.next_msg(timeout=30.0)  # 30 second timeout
            await sub.unsubscribe()
            
            # Parse and return result
            result = json.loads(msg.data.decode())
            
            # Log audit event
            await log_audit(
                discord_user["user_id"], 
                f"job_{job_type}", 
                None, 
                request, 
                {"job_payload": job_payload, "result_status": result.get("status")}
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Job processing timeout or error: {e}")
            return {
                "status": "error", 
                "error": "Job processing timeout or failed",
                "details": str(e)
            }
            
    except Exception as e:
        logging.error(f"Job submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Public health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/status")
@rate_limit(max_requests=5, window_minutes=1)
async def status(
    request: Request,
    _api_secret: bool = Depends(verify_api_secret),
    discord_user: dict = Depends(verify_discord_user)
):
    services = {}

    # Check NATS
    services['nats'] = 'connected' if nc and nc.is_connected else 'disconnected'

    # Check PostgreSQL
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        services['postgresql'] = 'healthy'
    except:
        services['postgresql'] = 'unhealthy'

    # Check Qdrant
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.getenv('QDRANT_URL')}/",
                headers={"api-key": os.getenv("QDRANT_API_KEY")},
                timeout=2
            )
        services['qdrant'] = 'healthy'
    except:
        services['qdrant'] = 'unhealthy'

    # Check S3
    try:
        s3.list_buckets()
        services['storage'] = 'healthy'
    except:
        services['storage'] = 'unhealthy'

    # Check Redis
    try:
        redis_client.ping()
        services['cache'] = 'healthy'
    except:
        services['cache'] = 'unhealthy'

    await log_audit(discord_user["user_id"], "status_check", None, request)
    return {"status": "operational", "services": services}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT == "development"
    )