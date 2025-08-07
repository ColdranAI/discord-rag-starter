# Discord RAG Bot Usage Examples

## 🚀 **Command Examples**

### **Text Commands**

```bash
# Upload a document
!add
# (Attach PDF, TXT, or MD file to the message)

# Basic search (shows search results with reaction buttons)
!ask What is the main topic of the documents?

# Search with specific LLM providers
!ask openai How does authentication work?
!ask gemini What are the security best practices?
!ask anthropic Summarize the deployment process
!ask gpt Explain the error handling strategy
!ask google What are the performance optimizations?

# Document management
!list                    # List your documents
!delete doc-uuid-123     # Delete a document
!status                  # Check system health
!help                    # Show help
```

### **Slash Commands**

```bash
# Basic search
/ask query: What is the API rate limiting strategy?

# Search with LLM provider
/ask query: How does caching work? provider: openai
/ask query: Explain the vector embeddings provider: claude
```

### **Interactive Usage**

1. **Basic Search**: Use `!ask <question>` to get search results
2. **AI Enhancement**: React with 🤖 (OpenAI), 🧠 (Gemini), or 🔬 (Claude) 
3. **Direct AI**: Use `!ask openai <question>` for immediate AI response

## 📚 **Document Processing Examples**

### **PDF Processing Example**

```python
# Example: Processing a technical documentation PDF

# Input: 50-page API documentation PDF
# Output: 
# - 45 semantic chunks (avg 1000 tokens each)
# - 384-dimensional embeddings per chunk
# - Metadata: filename, page numbers, sections
# - Stored in Qdrant with user-specific filtering

# Query example:
# !ask openai "How do I authenticate API requests?"
# 
# Result:
# - Retrieves top 5 relevant chunks about authentication
# - Generates context-aware prompt with source citations
# - Returns comprehensive answer with source references
```

### **Text File Processing**

```python
# Example: Processing code documentation

# Input: README.md, CONTRIBUTING.md, API_GUIDE.md
# Output:
# - Intelligent chunking by headers and paragraphs
# - Preserved formatting and code blocks
# - Cross-referenced sections
# - Searchable by functionality, setup steps, examples

# Query example:
# !ask anthropic "What are the environment variables needed?"
#
# Result: Lists all env vars with descriptions and examples
```

## 🔧 **Configuration Examples**

### **Environment Variables for Different Use Cases**

#### **Cost-Optimized Setup (GPT-3.5)**
```bash
USE_OPENAI=true
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-3.5-turbo

# Smaller chunks for faster processing
CHUNK_SIZE=800
RETRIEVAL_K=3
CACHE_TTL=7200  # 2 hours cache

# Lower costs: ~$0.002 per query
```

#### **Quality-Focused Setup (Claude Sonnet)**
```bash
USE_ANTHROPIC=true
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Larger chunks for better context
CHUNK_SIZE=1500
RETRIEVAL_K=5
CACHE_TTL=3600  # 1 hour cache

# Higher quality: ~$0.015 per query
```

#### **Multi-Provider Setup**
```bash
USE_OPENAI=true
USE_GEMINI=true
USE_ANTHROPIC=true

# Different models for different use cases:
# OpenAI: Quick answers, summarization
# Gemini: Complex reasoning, math
# Claude: Detailed analysis, creative tasks
```

## 📊 **Performance Examples**

### **Typical Processing Times**

```
Document Upload (10MB PDF):
├── Text extraction: 2-5 seconds
├── Chunking: 1-2 seconds  
├── Embedding generation: 5-15 seconds
├── Qdrant storage: 1-3 seconds
└── Total: 9-25 seconds

Query Processing:
├── Vector search: 100-300ms
├── LLM call (OpenAI): 1-3 seconds
├── LLM call (Claude): 2-5 seconds  
├── LLM call (Gemini): 1-4 seconds
└── Response formatting: 50-100ms

Cache Hit: 50-150ms total
```

### **Scaling Characteristics**

```
Storage Capacity:
├── 1,000 documents: ~500MB vectors, 2GB text
├── 10,000 documents: ~5GB vectors, 20GB text
├── Query time: O(log n) - scales logarithmically

Cost Analysis (monthly):
├── Infrastructure: $15-50 (Zerops)
├── LLM API: $10-100 (usage-based)
├── Storage: $5-20 (documents + vectors)
└── Total: $30-170/month
```

## 🛠️ **Advanced Usage Patterns**

### **Batch Document Processing**

```python
# Process multiple documents efficiently:

# 1. Upload multiple files in sequence
!add  # Upload doc1.pdf
!add  # Upload doc2.pdf  
!add  # Upload doc3.pdf

# 2. Wait for processing (check status)
!status

# 3. Query across all documents
!ask openai "Compare the approaches mentioned in these documents"
```

### **Context-Aware Conversations**

```python
# Build conversational context using caching:

# 1. Ask initial question
!ask anthropic "What is the system architecture?"

# 2. Follow-up questions leverage cache
!ask anthropic "How does the authentication in this architecture work?"
!ask anthropic "What are potential security issues with this approach?"

# Cache ensures consistent context across questions
```

### **Source Verification**

```python
# Always check sources in AI responses:

# Query: "What is the recommended database setup?"
# Response includes:
# - AI-generated answer
# - Source documents with scores
# - Relevant text excerpts
# - Confidence indicators

# Verify by asking for specific sources:
!ask "Show me the exact quote about database configuration"
```

## 🎯 **Best Practices**

### **Document Organization**

1. **Use descriptive filenames**: `api-authentication-guide.pdf`
2. **Structure content clearly**: Use headers, bullet points
3. **Keep documents focused**: One topic per document
4. **Update regularly**: Delete outdated documents

### **Query Optimization**

1. **Be specific**: "How to configure Redis caching?" vs "Redis?"
2. **Use context**: "In the deployment guide, what are the requirements?"
3. **Ask follow-ups**: Build on previous questions for context
4. **Choose the right LLM**: 
   - OpenAI: Quick factual answers
   - Claude: Detailed analysis
   - Gemini: Complex reasoning

### **Cost Management**

1. **Use caching**: Identical queries are cached
2. **Optimize chunks**: Balance context vs cost
3. **Monitor usage**: Check logs for expensive queries
4. **Choose models wisely**: GPT-3.5 vs GPT-4 vs Claude

## 🔍 **Debugging Examples**

### **No Results Found**

```bash
# Problem: !ask "project setup" returns no results

# Solutions:
1. Check if documents are processed: !status
2. Try broader terms: !ask "installation configuration"
3. List documents: !list
4. Upload relevant docs: !add (with setup guide)
```

### **Poor LLM Responses**

```bash
# Problem: AI gives generic responses

# Solutions:
1. Upload more specific documents
2. Ask more detailed questions
3. Try different LLM providers
4. Check if documents contain the information
```

### **Performance Issues**

```bash
# Problem: Slow responses

# Check:
1. System status: !status
2. Document count: !list
3. Query complexity
4. Network connectivity

# Optimize:
1. Reduce CHUNK_SIZE for faster embedding
2. Lower RETRIEVAL_K for fewer chunks
3. Use cache-friendly queries
```

## 📈 **Monitoring and Analytics**

### **Performance Metrics**

```sql
-- Query performance by LLM provider
SELECT 
    llm_provider,
    AVG(processing_time) as avg_time,
    COUNT(*) as query_count,
    AVG(token_count) as avg_tokens
FROM llm_interactions 
WHERE timestamp > NOW() - INTERVAL '24 HOURS'
GROUP BY llm_provider;

-- Most common queries
SELECT 
    LEFT(query, 50) as query_preview,
    COUNT(*) as frequency,
    AVG(processing_time) as avg_time
FROM llm_interactions 
GROUP BY LEFT(query, 50)
ORDER BY frequency DESC
LIMIT 10;

-- User activity patterns
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as queries,
    COUNT(DISTINCT user_id) as active_users
FROM llm_interactions 
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

This comprehensive setup gives you a production-ready, cost-effective, and highly capable Discord RAG bot that leverages the best of modern LLM technology while maintaining security and performance! 🚀
