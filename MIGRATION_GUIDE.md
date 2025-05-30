# Migration Guide: OpenAI to OpenRouter

This guide helps you migrate from using OpenAI for everything to using OpenRouter for text generation and OpenAI for embeddings.

## What Changed

### Before (OpenAI only)
- All API calls went to OpenAI
- Single API key: `OPENAI_API_KEY`
- Model: `gpt-4o-mini`

### After (OpenRouter + OpenAI)
- Text generation: OpenRouter
- Embeddings: OpenAI
- Two API keys: `OPENROUTER_API_KEY` and `OPENAI_API_KEY`
- Model: `openai/gpt-4o-mini` (note the prefix)

## Migration Steps

### 1. Get an OpenRouter API Key

1. Go to https://openrouter.ai/
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Add credits to your account

### 2. Update Environment Variables

```bash
# Keep your existing OpenAI key for embeddings
export OPENAI_API_KEY="your-existing-openai-key"

# Add the new OpenRouter key
export OPENROUTER_API_KEY="your-new-openrouter-key"
```

### 3. Test Your Setup

Run the test script to verify everything works:

```bash
python test_openrouter.py
```

### 4. Run Your Analysis

No changes needed to your command:

```bash
python analyze_tasks.py
```

## Benefits of This Change

1. **Model Variety**: Access to 200+ models from various providers
2. **Cost Optimization**: Choose cheaper models or use free tiers
3. **Better Uptime**: Automatic fallbacks between providers
4. **Same API**: OpenRouter uses OpenAI-compatible API

## Cost Comparison

| Task | Before (OpenAI) | After (OpenRouter) |
|------|-----------------|-------------------|
| Text Generation | $0.15/1M tokens | $0.15/1M tokens (same) or less with other models |
| Embeddings | $0.13/1M tokens | $0.13/1M tokens (unchanged) |

## Customization Options

### Use a Different Model

Edit `constants.py`:

```python
# Try Claude
OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"

# Try a free model
OPENROUTER_MODEL: str = "google/gemini-2.0-flash-exp:free"

# Use the cheapest available provider
OPENROUTER_MODEL: str = "openai/gpt-4o-mini:floor"
```

### Model Recommendations

For similar quality to GPT-4o-mini but potentially cheaper:
- `anthropic/claude-3.5-haiku` - Fast and efficient
- `google/gemini-2.0-flash-exp` - Google's efficient model
- `meta-llama/llama-3.1-8b-instruct` - Open source alternative

For higher quality:
- `anthropic/claude-3.5-sonnet` - Excellent reasoning
- `openai/gpt-4o` - Latest GPT-4
- `google/gemini-pro-1.5` - Google's flagship

## Rollback Plan

If you need to go back to OpenAI-only:

1. Change the imports back in `extractor.py` and `clustering.py`
2. Remove the OpenRouter-specific code
3. Use only `OPENAI_API_KEY`

But we recommend trying OpenRouter as it provides more flexibility! 