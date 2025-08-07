# Discord RAG Bot with LLM Integration

A **Discord bot with RAG (Retrieval Augmented Generation) capabilities** powered by **OpenAI, Google Gemini, and Anthropic Claude** Upload documents and get AI-powered responses through Discord commands with enterprise-grade security.

ps: if you find any flaw feel free to file an issue or pr :D

## What This Example Shows

Deploy 7 interconnected services in 60 seconds:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Qdrant    │     │ PostgreSQL  │     │   Valkey    │
│ (vectors)   │     │ (metadata)  │     │  (cache)    │
└─────────────┘     └─────────────┘     └─────────────┘
       ↑                    ↑                    ↑
       └────────────────────┼────────────────────┘
                           │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  FastAPI    │────▶│    NATS     │────▶│   Worker    │
│   (API)     │     │  (queue)    │     │ (processor) │
└─────────────┘     └─────────────┘     └─────────────┘
       ↑                                         │
       │                                         ▼
┌─────────────┐                         ┌─────────────┐
│ Discord Bot │                         │     S3      │
│  (client)   │                         │  (storage)  │
└─────────────┘                         └─────────────┘
```


## Discord Commands

### Document Management
- `!add` / `!upload` - Upload a document (attach file to message)
- `!list` / `!docs` - List your uploaded documents  
- `!delete <id>` - Delete a document by ID

### Search & AI Commands
- `!ask <query>` - Search documents (react with 🤖🧠🔬 for AI enhancement)
- `!openai <query>` / `!gpt <query>` - Search with OpenAI GPT
- `!gemini <query>` / `!google <query>` - Search with Google Gemini
- `!claude <query>` / `!anthropic <query>` - Search with Anthropic Claude

### System Commands
- `!status` - Check system health
- `!help` - Show help message

### How to Use AI Features
1. **Basic Search**: Use `!ask <your question>` to search your documents
2. **AI Enhancement**: React to search results with:
   - 🤖 for OpenAI GPT analysis
   - 🧠 for Google Gemini insights  
   - 🔬 for Anthropic Claude responses
3. **Direct AI Search**: Use `!openai`, `!gemini`, or `!claude` for immediate AI-enhanced responses
