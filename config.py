import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key, gotten from the .env file in the main folder. Modify it before use!
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# List of models - use OpenRouter model identifiers
MODEL_LIST = ['openai/gpt-5.2',
              'anthropic/claude-sonnet-4.5',
              'google/gemini-3-flash-preview',
              'deepseek/deepseek-v3.2',
              'meta-llama/llama-3.3-70b-instruct',
              'amazon/nova-2-lite-v1',
              'x-ai/grok-4.1-fast']

# Directories to save queries, graphs and parsing error logs
GRAPHS = "html_files"
HISTORY_CONVS = "data/queries"
HISTORY_LOGS = "data/logs"
