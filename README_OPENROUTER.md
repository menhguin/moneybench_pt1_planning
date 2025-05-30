# Moneybench Task Analysis with OpenRouter

This project has been updated to use OpenRouter for text generation while keeping OpenAI for embeddings.

## Setup

### 1. Install Dependencies

```bash
pip install openai scikit-learn numpy matplotlib
```

### 2. Set Environment Variables

You need two API keys:

```bash
# For embeddings (OpenAI)
export OPENAI_API_KEY="your-openai-api-key"

# For text generation (OpenRouter)
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

You can get an OpenRouter API key from: https://openrouter.ai/

### 3. Configure Models (Optional)

Edit `constants.py` to change the models:

```python
# OpenRouter model for text generation
# Examples of available models:
OPENROUTER_MODEL: str = "openai/gpt-4o-mini"  # Default
# OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"
# OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
# OPENROUTER_MODEL: str = "google/gemini-2.0-flash-exp:free"

# OpenAI embedding model
EMBED_MODEL: str = "text-embedding-3-large"
```

## Usage

Run the analysis:

```bash
python analyze_tasks.py
```

This will:
1. Generate responses using your chosen model via OpenRouter
2. Extract and embed tasks using OpenAI embeddings
3. Cluster similar tasks
4. Generate visualizations and reports

## Cost Optimization

OpenRouter offers several ways to optimize costs:

1. **Free models**: Add `:free` suffix (e.g., `"google/gemini-2.0-flash-exp:free"`)
2. **Price optimization**: Use `:floor` suffix to route to cheapest provider
3. **Speed optimization**: Use `:nitro` suffix for fastest response times

Example:
```python
OPENROUTER_MODEL: str = "openai/gpt-4o-mini:floor"  # Cheapest provider
```

## Supported Models

See all available models at: https://openrouter.ai/models

Popular options:
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/o1-mini`
- Anthropic: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3.5-haiku`
- Google: `google/gemini-2.0-flash-exp`, `google/gemini-pro-1.5`
- Meta: `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.1-70b-instruct`
- And many more...

## Why This Setup?

- **OpenRouter for text generation**: Access to multiple models, automatic fallbacks, unified pricing
- **OpenAI for embeddings**: High-quality embeddings that work well for clustering similar tasks

## Troubleshooting

If you get API key errors:
- Make sure both environment variables are set
- Check that your OpenRouter account has credits
- Verify your OpenAI API key has access to embeddings

For rate limiting issues:
- The code includes built-in delays
- You can increase delays in the code if needed
- Consider using OpenRouter's `:nitro` variant for better throughput 