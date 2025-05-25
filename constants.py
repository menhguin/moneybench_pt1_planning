from __future__ import annotations

"""Project-wide constants used by the Moneybench task-analysis scripts."""

MODEL: str = "gpt-4o-mini"
EMBED_MODEL: str = "text-embedding-3-large"

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