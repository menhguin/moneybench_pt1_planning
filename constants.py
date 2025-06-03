from __future__ import annotations

"""Project-wide constants used by the Moneybench task-analysis scripts."""

# OpenRouter model for text generation
# You can use any model from https://openrouter.ai/models
# Examples: "openai/gpt-4o-mini", "anthropic/claude-sonnet-4", "meta-llama/llama-3.1-8b-instruct", "google/gemini-2.5-pro-preview"
OPENROUTER_MODEL: str = "google/gemini-2.5-pro-preview"	

# OpenAI embedding model (keeping this on OpenAI)
EMBED_MODEL: str = "text-embedding-3-large"

# Legacy constant for backward compatibility
MODEL: str = OPENROUTER_MODEL

# How many independent queries of the LLM to collect per run
NUM_RUNS: int = 20

# How many clusters to group tasks into
N_CLUSTERS: int = 10

# Clustering parameters
MIN_CLUSTER_SIZE: int = 5  # Minimum tasks per cluster
MAX_CLUSTER_RATIO: float = 0.15  # Maximum ratio of clusters to total tasks

# Visual defaults
FIGSIZE: tuple[int, int] = (18, 14)

OUTPUT_DIR: str = "output"
ANALYSIS_DIR: str = "analysis" 