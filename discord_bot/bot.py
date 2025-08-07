import os
import asyncio
import aiohttp
import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
ADMIN_DISCORD_ID = int(os.getenv("ADMIN_DISCORD_ID")) if os.getenv("ADMIN_DISCORD_ID") else None
ALLOWED_GUILD_ID = int(os.getenv("ALLOWED_GUILD_ID")) if os.getenv("ALLOWED_GUILD_ID") else None
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
ALLOWED_FILE_EXTENSIONS = os.getenv("ALLOWED_FILE_EXTENSIONS", ".pdf,.txt,.md").split(",")

# Channel monitoring configuration
USE_CHANNEL = os.getenv("USE_CHANNEL", "false").lower() == "true"
CHANNEL_ID = int(os.getenv("CHANNEL_ID")) if os.getenv("CHANNEL_ID") else None
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discord_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Bot setup with minimal intents for security
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Add slash command tree
tree = bot.tree

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

def check_admin_user():
    """Decorator to check if user is the admin"""
    def predicate(ctx):
        if ctx.author.id != ADMIN_DISCORD_ID:
            raise SecurityError(f"Unauthorized user: {ctx.author.id}")
        return True
    return commands.check(predicate)

def check_allowed_guild():
    """Decorator to check if command is used in allowed guild"""
    def predicate(ctx):
        if ALLOWED_GUILD_ID and ctx.guild and ctx.guild.id != ALLOWED_GUILD_ID:
            raise SecurityError(f"Unauthorized guild: {ctx.guild.id}")
        return True
    return commands.check(predicate)

def check_dm_or_guild():
    """Decorator to ensure command is used in DM or allowed guild"""
    def predicate(ctx):
        if ctx.guild is None:  # DM
            return True
        if ALLOWED_GUILD_ID and ctx.guild.id == ALLOWED_GUILD_ID:
            return True
        raise SecurityError("Command must be used in DM or authorized guild")
    return commands.check(predicate)

class SecureAPIClient:
    """Secure API client for communicating with FastAPI backend"""
    
    def __init__(self, base_url: str, api_secret: str):
        self.base_url = base_url.rstrip('/')
        self.api_secret = api_secret
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Discord-RAG-Bot/1.0',
                'X-API-Secret': self.api_secret
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self, user_id: str, guild_id: Optional[str] = None) -> dict:
        """Get headers with Discord user information"""
        headers = {
            'X-Discord-User-ID': str(user_id),
            'X-API-Secret': self.api_secret
        }
        if guild_id:
            headers['X-Discord-Guild-ID'] = str(guild_id)
        return headers
    
    async def upload_file(self, file_data: bytes, filename: str, user_id: str, guild_id: Optional[str] = None) -> dict:
        """Upload file to API"""
        headers = self._get_headers(user_id, guild_id)
        
        data = aiohttp.FormData()
        data.add_field('file', file_data, filename=filename, content_type='application/octet-stream')
        
        async with self.session.post(
            f"{self.base_url}/upload",
            data=data,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Upload failed: {response.status} - {error_text}")
    
    async def search(self, query: str, user_id: str, guild_id: Optional[str] = None) -> dict:
        """Search documents"""
        headers = self._get_headers(user_id, guild_id)
        
        async with self.session.get(
            f"{self.base_url}/search",
            params={'query': query},
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Search failed: {response.status} - {error_text}")
    
    async def search_with_llm(self, query: str, llm_provider: str, user_id: str, guild_id: Optional[str] = None) -> dict:
        """Search documents with LLM enhancement"""
        headers = self._get_headers(user_id, guild_id)
        
        async with self.session.get(
            f"{self.base_url}/search",
            params={'query': query, 'llm': llm_provider},
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"LLM search failed: {response.status} - {error_text}")
    
    async def submit_job(self, job_type: str, data: dict, llm_provider: str = None, user_id: str = None, guild_id: str = None) -> dict:
        """Submit job directly to NATS queue"""
        headers = self._get_headers(user_id, guild_id)
        
        payload = {
            'job_type': job_type,
            'data': data,
            'llm_provider': llm_provider
        }
        
        async with self.session.post(
            f"{self.base_url}/submit_job",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Job submission failed: {response.status} - {error_text}")
    
    async def list_documents(self, user_id: str, guild_id: Optional[str] = None) -> dict:
        """List user's documents"""
        headers = self._get_headers(user_id, guild_id)
        
        async with self.session.get(
            f"{self.base_url}/documents",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"List documents failed: {response.status} - {error_text}")
    
    async def delete_document(self, doc_id: str, user_id: str, guild_id: Optional[str] = None) -> dict:
        """Delete a document"""
        headers = self._get_headers(user_id, guild_id)
        
        async with self.session.delete(
            f"{self.base_url}/documents/{doc_id}",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Delete failed: {response.status} - {error_text}")
    
    async def get_status(self, user_id: str, guild_id: Optional[str] = None) -> dict:
        """Get system status"""
        headers = self._get_headers(user_id, guild_id)
        
        async with self.session.get(
            f"{self.base_url}/status",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Status check failed: {response.status} - {error_text}")

def validate_file(attachment: discord.Attachment) -> bool:
    """Validate file before upload"""
    # Check file size
    if attachment.size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
    
    # Check file extension
    file_ext = os.path.splitext(attachment.filename)[1].lower()
    if file_ext not in ALLOWED_FILE_EXTENSIONS:
        raise ValueError(f"File type not allowed. Allowed extensions: {', '.join(ALLOWED_FILE_EXTENSIONS)}")
    
    # Check filename for security
    if ".." in attachment.filename or "/" in attachment.filename or "\\" in attachment.filename:
        raise ValueError("Invalid filename")
    
    return True

@bot.event
async def on_ready():
    """Bot startup event"""
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guilds')
    
    # Validate configuration
    if not API_SECRET_KEY:
        logger.error("API_SECRET_KEY not configured!")
        await bot.close()
        return
    
    if not ADMIN_DISCORD_ID:
        logger.error("ADMIN_DISCORD_ID not configured!")
        await bot.close()
        return
    
    # Channel monitoring setup
    if USE_CHANNEL:
        if CHANNEL_ID:
            logger.info(f"Channel monitoring enabled for channel ID: {CHANNEL_ID}")
        else:
            logger.warning("USE_CHANNEL is true but CHANNEL_ID not set!")
    
    # Sync slash commands
    try:
        synced = await tree.sync()
        logger.info(f"Synced {len(synced)} slash commands")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}")
    
    # Test API connection
    try:
        async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
            async with client.session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    logger.info("API connection test successful")
                else:
                    logger.warning(f"API health check returned: {response.status}")
    except Exception as e:
        logger.error(f"Failed to connect to API: {e}")

@bot.event
async def on_command_error(ctx, error):
    """Global error handler"""
    if isinstance(error, commands.CheckFailure):
        if isinstance(error.original, SecurityError):
            logger.warning(f"Security violation: {error.original} - User: {ctx.author.id}, Guild: {ctx.guild.id if ctx.guild else 'DM'}")
            await ctx.send("❌ You are not authorized to use this command.")
        else:
            await ctx.send("❌ You don't have permission to use this command.")
    elif isinstance(error, commands.CommandNotFound):
        # Ignore unknown commands for security
        pass
    else:
        logger.error(f"Command error: {error}")
        await ctx.send("❌ An error occurred while processing your command.")

@bot.event
async def on_reaction_add(reaction, user):
    """Handle reaction-based LLM requests"""
    # Ignore bot reactions
    if user.bot:
        return
    
    # Check if user is authorized
    if user.id != ADMIN_DISCORD_ID:
        return
    
    # Check if reaction is on bot's message with search results
    if reaction.message.author != bot.user:
        return
    
    # Check if message has embeds with search results
    if not reaction.message.embeds:
        return
    
    embed = reaction.message.embeds[0]
    if "Search Results" not in embed.title:
        return
    
    # Extract query from embed description
    description = embed.description
    if not description or "**Query:**" not in description:
        return
    
    query = description.split("**Query:**")[1].split("\n")[0].strip()
    
    # Map reactions to LLM providers
    llm_map = {
        '🤖': ('openai', '🤖 OpenAI GPT'),
        '🧠': ('gemini', '🧠 Google Gemini'),
        '🔬': ('anthropic', '🔬 Anthropic Claude')
    }
    
    if str(reaction.emoji) in llm_map:
        llm_provider, provider_name = llm_map[str(reaction.emoji)]
        
        try:
            # Remove user's reaction
            await reaction.remove(user)
            
            # Create a temporary context-like object for the LLM search
            class TempCtx:
                def __init__(self, channel, author, guild):
                    self.channel = channel
                    self.author = author
                    self.guild = guild
                    self.send = channel.send
            
            temp_ctx = TempCtx(reaction.message.channel, user, reaction.message.guild)
            
            # Perform LLM search
            await search_with_llm(temp_ctx, query, llm_provider, provider_name)
            
        except Exception as e:
            logger.error(f"Reaction LLM search error: {e}")
            await reaction.message.channel.send(f"❌ **{provider_name} search failed:** {str(e)}")

@bot.event
async def on_message(message):
    """Handle messages in monitored channel and mentions"""
    # Ignore bot messages
    if message.author.bot:
        return
    
    # Check if this is a command first
    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.process_commands(message)
        return
    
    should_respond = False
    is_admin = message.author.id == ADMIN_DISCORD_ID
    
    # Check if bot is mentioned
    if bot.user in message.mentions:
        should_respond = True
        logger.info(f"Bot mentioned by {message.author.id} in {message.channel.id}")
    
    # Check if in monitored channel (any user can ask)
    elif USE_CHANNEL and CHANNEL_ID and message.channel.id == CHANNEL_ID:
        should_respond = True
        logger.info(f"Message in monitored channel by {message.author.id}")
    
    # Check if admin user triggered with !ask in any channel
    elif is_admin and message.content.startswith("!ask"):
        should_respond = True
        logger.info(f"Admin triggered !ask command in {message.channel.id}")
    
    if should_respond:
        await handle_natural_query(message, is_admin)
    else:
        # Process other commands normally
        await bot.process_commands(message)

async def handle_natural_query(message, is_admin_user=False):
    """Handle natural language queries from mentions or monitored channel"""
    try:
        # Extract the query from the message
        content = message.content
        
        # Remove bot mention if present
        if bot.user in message.mentions:
            # Remove mention from content
            content = content.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
        
        # Remove !ask if present (for admin commands)
        if content.startswith("!ask"):
            content = content[4:].strip()
        
        # Parse provider and options
        llm_provider = None
        use_web_search = False
        valid_providers = ["openai", "gemini", "claude", "anthropic", "gpt", "google"]
        
        # Check for web search flag
        if "web:" in content.lower() or content.lower().startswith("web "):
            use_web_search = True
            content = content.replace("web:", "").replace("Web:", "").strip()
            if content.lower().startswith("web "):
                content = content[4:].strip()
        
        # Check if message starts with a provider name
        words = content.split()
        if words and words[0].lower() in valid_providers:
            provider_word = words[0].lower()
            # Normalize provider names
            provider_map = {
                "gpt": "openai",
                "google": "gemini",
                "claude": "anthropic"
            }
            llm_provider = provider_map.get(provider_word, provider_word)
            content = " ".join(words[1:])  # Remove provider from query
        else:
            # Use default provider
            llm_provider = DEFAULT_LLM_PROVIDER
        
        # Validate query length
        if len(content.strip()) < 3:
            await message.reply("❓ Your question seems too short. Could you provide more details?")
            return
        
        if len(content) > 500:
            await message.reply("📝 Your question is quite long. Could you try to make it more concise? (max 500 characters)")
            return
        
        # Show typing indicator
        async with message.channel.typing():
            # Determine user permissions
            user_id = ADMIN_DISCORD_ID if is_admin_user else message.author.id
            guild_id = message.guild.id if message.guild else None
            
            # Submit query job
            async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
                result = await client.submit_job(
                    "query",
                    {"query": content, "use_web_search": use_web_search},
                    llm_provider,
                    user_id,
                    guild_id
                )
            
            # Format response
            if result.get('status') == 'success' and result.get('response'):
                # Create embed response
                provider_emojis = {"openai": "🤖", "gemini": "🧠", "anthropic": "🔬"}
                provider_names = {"openai": "OpenAI", "gemini": "Gemini", "anthropic": "Claude"}
                
                embed = discord.Embed(
                    title=f"{provider_emojis.get(llm_provider, '🤖')} {provider_names.get(llm_provider, llm_provider.title())} Response",
                    description=content[:100] + "..." if len(content) > 100 else content,
                    color=0x2ecc71,
                    timestamp=datetime.utcnow()
                )
                
                # Add main response
                response_text = result['response']
                if len(response_text) > 1024:
                    response_text = response_text[:1021] + "..."
                
                embed.add_field(
                    name="Answer",
                    value=response_text,
                    inline=False
                )
                
                # Add sources if available
                sources_text = ""
                if result.get('sources'):
                    doc_sources = [f"📄 {s['filename']}" for s in result['sources'][:3]]
                    if doc_sources:
                        sources_text += "**Documents:** " + ", ".join(doc_sources)
                
                if result.get('web_sources'):
                    web_sources = [f"🌐 [{s['title']}]({s['url']})" for s in result['web_sources'][:2]]
                    if web_sources:
                        if sources_text:
                            sources_text += "\n"
                        sources_text += "**Web:** " + ", ".join(web_sources)
                
                if sources_text:
                    embed.add_field(
                        name="Sources",
                        value=sources_text,
                        inline=False
                    )
                
                # Add footer info
                footer_parts = []
                if result.get('cached'):
                    footer_parts.append("⚡ Cached")
                if result.get('used_web_search'):
                    footer_parts.append("🌐 Web Enhanced")
                if not is_admin_user:
                    footer_parts.append("👥 Public Channel")
                
                if footer_parts:
                    embed.set_footer(text=" • ".join(footer_parts))
                
                await message.reply(embed=embed)
                
            else:
                # Error or no response
                error_msg = result.get('error', 'Unable to process your question')
                embed = discord.Embed(
                    title="❌ Error",
                    description=f"Sorry, I couldn't process your question: {error_msg}",
                    color=0xe74c3c
                )
                await message.reply(embed=embed)
        
        logger.info(f"Natural query processed for user {message.author.id}: '{content[:50]}...' with {llm_provider}")
        
    except Exception as e:
        logger.error(f"Natural query processing error: {e}")
        await message.reply("❌ Sorry, I encountered an error processing your question. Please try again later.")

@bot.command(name='add', aliases=['upload'])
@check_admin_user()
@check_dm_or_guild()
async def upload_document(ctx):
    """Upload a document for RAG processing"""
    if not ctx.message.attachments:
        embed = discord.Embed(
            title="📎 Upload Document",
            description="Please attach a file to upload.\n\n**Supported formats:** " + ", ".join(ALLOWED_FILE_EXTENSIONS) + f"\n**Max size:** {MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
            color=0x3498db
        )
        await ctx.send(embed=embed)
        return
    
    attachment = ctx.message.attachments[0]
    
    try:
        # Validate file
        validate_file(attachment)
        
        # Show processing message
        processing_msg = await ctx.send("⏳ Uploading and processing document...")
        
        # Download file
        file_data = await attachment.read()
        
        # Upload via API
        async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
            result = await client.upload_file(
                file_data, 
                attachment.filename, 
                ctx.author.id,
                ctx.guild.id if ctx.guild else None
            )
        
        # Success response
        embed = discord.Embed(
            title="✅ Document Uploaded Successfully",
            description=f"**File:** {attachment.filename}\n**ID:** `{result['id']}`\n**Status:** {result['status']}",
            color=0x2ecc71,
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="Document is being processed for search")
        
        await processing_msg.edit(content="", embed=embed)
        
        logger.info(f"Document uploaded successfully: {result['id']} by user {ctx.author.id}")
        
    except ValueError as e:
        await ctx.send(f"❌ **File validation error:** {e}")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        await ctx.send(f"❌ **Upload failed:** {str(e)}")

# Slash command for asking with LLM provider selection
@tree.command(name="ask", description="Ask a question with optional AI provider")
async def slash_ask(interaction: discord.Interaction, query: str, provider: str = None):
    """Slash command for asking questions"""
    # Check authorization
    if interaction.user.id != ADMIN_DISCORD_ID:
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return
    
    # Validate provider
    valid_providers = ["openai", "gemini", "claude", None]
    if provider and provider.lower() not in valid_providers:
        await interaction.response.send_message("❌ Invalid provider. Use: openai, gemini, or claude", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        await process_ask_command(interaction, query, provider.lower() if provider else None)
    except Exception as e:
        logger.error(f"Slash ask error: {e}")
        await interaction.followup.send(f"❌ **Error:** {str(e)}")

@bot.command(name='ask')
@check_admin_user()
@check_dm_or_guild()
async def ask_command(ctx, provider: str = None, *, query: str = None):
    """Ask command with provider selection: !ask [provider] <question>"""
    # Parse arguments - if provider is not a valid provider, treat it as part of the query
    valid_providers = ["openai", "gemini", "claude", "anthropic", "gpt", "google"]
    
    if provider and provider.lower() in valid_providers:
        # Provider specified
        if not query:
            await ctx.send("❌ Please provide a question after the provider.")
            return
        
        # Normalize provider names
        provider_map = {
            "gpt": "openai",
            "google": "gemini",
            "claude": "anthropic"
        }
        llm_provider = provider_map.get(provider.lower(), provider.lower())
    else:
        # No provider specified, treat first argument as part of query
        if query:
            query = f"{provider} {query}"
        else:
            query = provider or ""
        llm_provider = None
    
    if not query or len(query.strip()) < 3:
        await ctx.send("❌ Query must be at least 3 characters long.")
        return
    
    try:
        await process_ask_command(ctx, query, llm_provider)
    except Exception as e:
        logger.error(f"Ask command error: {e}")
        await ctx.send(f"❌ **Error:** {str(e)}")

async def process_ask_command(ctx_or_interaction, query: str, llm_provider: str = None):
    """Process ask command for both slash and text commands"""
    is_interaction = hasattr(ctx_or_interaction, 'response')
    
    if len(query.strip()) < 3:
        message = "❌ Query must be at least 3 characters long."
        if is_interaction:
            await ctx_or_interaction.followup.send(message)
        else:
            await ctx_or_interaction.send(message)
        return
    
    if len(query) > 500:
        message = "❌ Query is too long (max 500 characters)."
        if is_interaction:
            await ctx_or_interaction.followup.send(message)
        else:
            await ctx_or_interaction.send(message)
        return
    
    # Show processing message
    if llm_provider:
        provider_names = {"openai": "🤖 OpenAI", "gemini": "🧠 Gemini", "anthropic": "🔬 Claude"}
        processing_text = f"🔍 Searching with {provider_names.get(llm_provider, llm_provider)}..."
    else:
        processing_text = "🔍 Searching documents..."
    
    if is_interaction:
        await ctx_or_interaction.followup.send(processing_text)
        user_id = ctx_or_interaction.user.id
        guild_id = ctx_or_interaction.guild_id
    else:
        processing_msg = await ctx_or_interaction.send(processing_text)
        user_id = ctx_or_interaction.author.id
        guild_id = ctx_or_interaction.guild.id if ctx_or_interaction.guild else None
    
    try:
        async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
            if llm_provider:
                # Submit job for LLM processing
                result = await client.submit_job(
                    "query",
                    {"query": query},
                    llm_provider,
                    user_id,
                    guild_id
                )
            else:
                # Regular search
                result = await client.search(query, user_id, guild_id)
        
        # Format response
        if llm_provider and result.get('response'):
            # LLM-enhanced response
            embed = discord.Embed(
                title=f"🤖 AI Response",
                description=f"**Query:** {query}",
                color=0x2ecc71,
                timestamp=datetime.utcnow()
            )
            
            # Add AI response
            response_text = result['response']
            if len(response_text) > 1000:
                response_text = response_text[:1000] + "..."
            
            embed.add_field(
                name=f"Response ({llm_provider.title()})",
                value=response_text,
                inline=False
            )
            
            # Add sources if available
            if result.get('sources'):
                sources_text = "\n".join([
                    f"• **{source['filename']}** (Score: {source['score']:.3f})"
                    for source in result['sources'][:3]
                ])
                embed.add_field(
                    name="📚 Sources",
                    value=sources_text,
                    inline=False
                )
            
            if result.get('cached'):
                embed.set_footer(text="⚡ Cached response")
        
        elif result.get('results'):
            # Regular search results
            embed = discord.Embed(
                title="🔍 Search Results",
                description=f"**Query:** {query}\n**Found:** {len(result['results'])} results",
                color=0x3498db,
                timestamp=datetime.utcnow()
            )
            
            for i, doc in enumerate(result['results'][:3], 1):
                payload = doc.get('payload', {})
                text_preview = payload.get('content', '')[:200] + "..." if len(payload.get('content', '')) > 200 else payload.get('content', '')
                
                embed.add_field(
                    name=f"📄 Result {i}: {payload.get('filename', 'Unknown')}",
                    value=f"**Score:** {doc.get('score', 0):.3f}\n**Preview:** {text_preview}",
                    inline=False
                )
            
            if not llm_provider:
                embed.set_footer(text="💡 Use !ask openai/gemini/claude <query> for AI-enhanced responses")
        
        else:
            # No results
            embed = discord.Embed(
                title="🔍 Search Results",
                description=f"**Query:** {query}\n\nNo results found.",
                color=0xe74c3c
            )
            embed.add_field(
                name="💡 Suggestions",
                value="• Try rephrasing your question\n• Upload relevant documents first\n• Check for typos",
                inline=False
            )
        
        # Send response
        if is_interaction:
            await ctx_or_interaction.edit_original_response(content="", embed=embed)
        else:
            await processing_msg.edit(content="", embed=embed)
            
            # Add reaction buttons for regular search
            if not llm_provider and result.get('results'):
                await processing_msg.add_reaction('🤖')  # OpenAI
                await processing_msg.add_reaction('🧠')  # Gemini
                await processing_msg.add_reaction('🔬')  # Anthropic
        
        logger.info(f"Query processed for user {user_id}: '{query}' with provider {llm_provider}")
        
    except Exception as e:
        logger.error(f"Ask processing error: {e}")
        error_msg = f"❌ **Processing failed:** {str(e)}"
        
        if is_interaction:
            await ctx_or_interaction.edit_original_response(content=error_msg)
        else:
            await processing_msg.edit(content=error_msg)

@bot.command(name='openai', aliases=['gpt'])
@check_admin_user()
@check_dm_or_guild()
async def search_with_openai(ctx, *, query: str):
    """Search documents with OpenAI-enhanced response"""
    await search_with_llm(ctx, query, "openai", "🤖 OpenAI GPT")

@bot.command(name='gemini', aliases=['google'])
@check_admin_user()
@check_dm_or_guild()
async def search_with_gemini(ctx, *, query: str):
    """Search documents with Gemini-enhanced response"""
    await search_with_llm(ctx, query, "gemini", "🧠 Google Gemini")

@bot.command(name='claude', aliases=['anthropic'])
@check_admin_user()
@check_dm_or_guild()
async def search_with_anthropic(ctx, *, query: str):
    """Search documents with Anthropic Claude-enhanced response"""
    await search_with_llm(ctx, query, "anthropic", "🔬 Anthropic Claude")

async def search_with_llm(ctx, query: str, llm_provider: str, provider_name: str):
    """Helper function to search with LLM enhancement"""
    if len(query.strip()) < 3:
        await ctx.send("❌ Query must be at least 3 characters long.")
        return
    
    if len(query) > 500:
        await ctx.send("❌ Query is too long (max 500 characters).")
        return
    
    try:
        # Show processing message
        processing_msg = await ctx.send(f"🔍 Searching documents with {provider_name}...")
        
        # Search via API with LLM
        async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
            result = await client.search_with_llm(
                query, 
                llm_provider,
                ctx.author.id,
                ctx.guild.id if ctx.guild else None
            )
        
        if not result.get('results'):
            embed = discord.Embed(
                title=f"🔍 {provider_name} Search Results",
                description=f"**Query:** {query}\n\nNo results found.",
                color=0xe74c3c
            )
        else:
            embed = discord.Embed(
                title=f"🔍 {provider_name} Enhanced Response",
                description=f"**Query:** {query}",
                color=0x2ecc71,
                timestamp=datetime.utcnow()
            )
            
            # Add LLM response if available
            if result.get('llm_response'):
                embed.add_field(
                    name=f"{provider_name} Response",
                    value=result['llm_response'][:1000] + ("..." if len(result['llm_response']) > 1000 else ""),
                    inline=False
                )
            
            # Add source documents
            if result.get('results'):
                sources = "\n".join([
                    f"• {doc.get('payload', {}).get('filename', 'Unknown')}"
                    for doc in result['results'][:3]
                ])
                embed.add_field(
                    name="📚 Sources",
                    value=sources,
                    inline=False
                )
        
        await processing_msg.edit(content="", embed=embed)
        
        logger.info(f"LLM search performed by user {ctx.author.id}: '{query}' with {llm_provider}")
        
    except Exception as e:
        logger.error(f"LLM search error: {e}")
        await ctx.send(f"❌ **{provider_name} search failed:** {str(e)}")

@bot.command(name='list', aliases=['docs', 'documents'])
@check_admin_user()
@check_dm_or_guild()
async def list_documents(ctx):
    """List your uploaded documents"""
    try:
        async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
            documents = await client.list_documents(
                ctx.author.id,
                ctx.guild.id if ctx.guild else None
            )
        
        if not documents:
            embed = discord.Embed(
                title="📚 Your Documents",
                description="No documents found. Use `!add` to upload your first document.",
                color=0xe74c3c
            )
        else:
            embed = discord.Embed(
                title="📚 Your Documents",
                description=f"Found {len(documents)} documents:",
                color=0x3498db,
                timestamp=datetime.utcnow()
            )
            
            for doc in documents[:10]:  # Limit to 10 documents
                status_emoji = "✅" if doc['processed'] else "⏳"
                upload_date = datetime.fromisoformat(doc['upload_date'].replace('Z', '+00:00'))
                
                embed.add_field(
                    name=f"{status_emoji} {doc['filename']}",
                    value=f"**ID:** `{doc['id']}`\n**Uploaded:** {upload_date.strftime('%Y-%m-%d %H:%M')}",
                    inline=True
                )
        
        await ctx.send(embed=embed)
        
        logger.info(f"Document list requested by user {ctx.author.id}")
        
    except Exception as e:
        logger.error(f"List documents error: {e}")
        await ctx.send(f"❌ **Failed to list documents:** {str(e)}")

@bot.command(name='delete', aliases=['remove', 'del'])
@check_admin_user()
@check_dm_or_guild()
async def delete_document(ctx, doc_id: str):
    """Delete a document by ID"""
    try:
        async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
            result = await client.delete_document(
                doc_id,
                ctx.author.id,
                ctx.guild.id if ctx.guild else None
            )
        
        embed = discord.Embed(
            title="🗑️ Document Deleted",
            description=f"**Document ID:** `{doc_id}`\n**Status:** {result['status']}",
            color=0xe74c3c,
            timestamp=datetime.utcnow()
        )
        
        await ctx.send(embed=embed)
        
        logger.info(f"Document deleted by user {ctx.author.id}: {doc_id}")
        
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        await ctx.send(f"❌ **Failed to delete document:** {str(e)}")

@bot.command(name='status', aliases=['health'])
@check_admin_user()
@check_dm_or_guild()
async def system_status(ctx):
    """Check system status"""
    try:
        async with SecureAPIClient(API_BASE_URL, API_SECRET_KEY) as client:
            status = await client.get_status(
                ctx.author.id,
                ctx.guild.id if ctx.guild else None
            )
        
        embed = discord.Embed(
            title="🔧 System Status",
            description=f"**Overall Status:** {status['status']}",
            color=0x2ecc71 if status['status'] == 'operational' else 0xe74c3c,
            timestamp=datetime.utcnow()
        )
        
        services = status.get('services', {})
        for service, health in services.items():
            emoji = "✅" if health == 'healthy' or health == 'connected' else "❌"
            embed.add_field(
                name=f"{emoji} {service.title()}",
                value=health.title(),
                inline=True
            )
        
        await ctx.send(embed=embed)
        
        logger.info(f"Status check by user {ctx.author.id}")
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        await ctx.send(f"❌ **Failed to get status:** {str(e)}")

@bot.command(name='help')
@check_admin_user()
@check_dm_or_guild()
async def help_command(ctx):
    """Show help information"""
    embed = discord.Embed(
        title="🤖 Discord RAG Bot Help",
        description="Secure document upload and search bot for personal use.",
        color=0x3498db,
        timestamp=datetime.utcnow()
    )
    
    commands_info = [
        ("!add / !upload", "Upload a document (attach file to message)"),
        ("!ask <query>", "Search documents (react with 🤖🧠🔬 for AI)"),
        ("!ask <provider> <query>", "Search with specific AI (openai/gemini/claude)"),
        ("@bot <query>", "Ask the bot naturally by mentioning it"),
        ("!list / !docs", "List your uploaded documents"),
        ("!delete <id>", "Delete a document by ID"),
        ("!status", "Check system health"),
        ("!help", "Show this help message")
    ]
    
    # Add channel-specific info
    if USE_CHANNEL and CHANNEL_ID:
        commands_info.insert(-3, ("🔴 Channel Mode", f"Bot listens to all messages in <#{CHANNEL_ID}>"))
    
    embed.add_field(
        name="💡 Natural Usage",
        value="• Mention the bot: `@bot what is machine learning?`\n"
              "• Use provider prefix: `@bot openai explain transformers`\n"
              f"• Default AI: {DEFAULT_LLM_PROVIDER.title()}\n"
              "• Web search included automatically 🌐",
        inline=False
    )
    
    for cmd, desc in commands_info:
        embed.add_field(name=cmd, value=desc, inline=False)
    
    embed.set_footer(text="This bot is restricted to authorized users only")
    
    await ctx.send(embed=embed)

async def main():
    """Main function to run the bot"""
    if not DISCORD_TOKEN:
        logger.error("DISCORD_BOT_TOKEN not found in environment variables")
        return
    
    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())