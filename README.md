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

### 2. Service Auto-Configuration

See how Zerops automatically wires services together using environment variables:

```python
# PostgreSQL connection - no manual config needed
db_pool = await asyncpg.create_pool(
    host=os.getenv("DB_HOST"),  # Zerops provides this
    password=os.getenv("DB_PASSWORD")  # And this
)

# Same for Redis, S3, NATS, Qdrant...
```

### 3. Build & Deploy Pipeline

The [`zerops.yml`](./zerops.yml) shows how to:
- Define build and runtime environments
- Configure auto-scaling
- Set up zero-downtime deployments

### 4. Development Workflow

Local development using cloud services:

```bash
# Connect to project's private network
zcli vpn up

# Get all service credentials
zcli env --dotenv > .env

# Your local code now uses Zerops databases
```

### 5. Production Patterns

- Async processing with queues
- Caching strategies  
- Service discovery
- High availability options

## Setup & Configuration

### 1. Deploy to Zerops
Click the deploy button above or import the project manually using `zerops-project-import.yml`.

### 2. Configure Discord Bot
After deployment, you need to set up these environment variables in Zerops GUI:

**For API service:**
- `API_SECRET_KEY` - Generate a random secret key
- `ADMIN_DISCORD_ID` - Your Discord user ID
- `ALLOWED_GUILD_ID` - Your Discord server ID (optional)
- `ENVIRONMENT` - Set to "production"

**LLM Configuration (choose one or more):**
- `USE_OPENAI` - Set to "true" to enable OpenAI
- `OPENAI_API_KEY` - Your OpenAI API key
- `OPENAI_MODEL` - Model to use (default: gpt-3.5-turbo)
- `USE_GEMINI` - Set to "true" to enable Google Gemini
- `GEMINI_API_KEY` - Your Google AI API key  
- `USE_ANTHROPIC` - Set to "true" to enable Anthropic Claude
- `ANTHROPIC_API_KEY` - Your Anthropic API key
- `ANTHROPIC_MODEL` - Model to use (default: claude-3-sonnet-20240229)

**For Discord Bot service:**
- `DISCORD_BOT_TOKEN` - Your Discord bot token
- `API_SECRET_KEY` - Same as API service
- `ADMIN_DISCORD_ID` - Same as API service
- `ALLOWED_GUILD_ID` - Same as API service (optional)

### 3. Create Discord Bot
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application and bot
3. Copy the bot token to `DISCORD_BOT_TOKEN`
4. Invite bot to your server with appropriate permissions

## Quick Demo

1. **Deploy**: Click the button above (60 seconds)
2. **Configure**: Set up your Discord bot token and user IDs in Zerops environment variables
3. **Upload**: Use `!add` command in Discord with file attachment
4. **Search**: Use `!ask <query>` to search your documents
5. **Explore**: Check logs, metrics, and service details in Zerops dashboard

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

## Key Files to Examine

| File | What It Demonstrates |
|------|---------------------|
| [`zerops-project-import.yml`](./zerops-project-import.yml) | Production environment with LLM configuration |
| [`zerops.yml`](./zerops.yml) | Build & deploy pipeline for all services |
| [`api/main.py`](./api/main.py) | Secure API with LLM integration (OpenAI, Gemini, Claude) |
| [`discord_bot/bot.py`](./discord_bot/bot.py) | Discord bot with AI-enhanced search commands |
| [`processor/processor.py`](./processor/processor.py) | Document processing and vector embeddings |
| [`env.example`](./env.example) | Complete environment variable configuration |

## Understanding the Integration

### Project Import Structure

```yaml
project:
  name: discord-rag-bot
  envVariables:
    # Zerops auto-references service credentials
    DB_HOST: ${db_hostname}
    QDRANT_URL: ${qdrant_connectionString}
    # ... automatic for all services

services:
  - hostname: db
    type: postgresql@16
    mode: NON_HA  # for development, use HA for production
  - hostname: discord-bot
    type: python@3.11
    envSecrets:
      DISCORD_BOT_TOKEN: ""
      API_SECRET_KEY: ""
      ADMIN_DISCORD_ID: ""
```

### Build Configuration

```yaml
# From zerops.yml
- setup: api
  build:
    base: python@3.11
    deployFiles: ./api/~
  run:
    base: python@3.11
    ports:
      - port: 8000
        httpSupport: true
    healthCheck:
      httpGet:
        port: 8000
        path: /health

- setup: discord-bot
  build:
    base: python@3.11
    deployFiles: ./discord_bot/~
  run:
    base: python@3.11
    envVariables:
      API_BASE_URL: http://api:8000
```

