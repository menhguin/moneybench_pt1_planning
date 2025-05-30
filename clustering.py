"""Clustering module for embedding and grouping similar tasks."""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from openai import AsyncOpenAI
from sklearn.cluster import KMeans

import constants as C


class TaskClusterer:
    """Handles embedding tasks and clustering them into groups."""
    
    def __init__(self, api_key: str | None = None):
        # Use OpenAI for embeddings
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable for embeddings")
        
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        
        # Use OpenRouter for chat completions (cluster naming)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("Please set OPENROUTER_API_KEY environment variable")
            
        self.openrouter_client = AsyncOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        self.embedding_model = C.EMBED_MODEL
    
    async def embed_tasks(self, tasks: List[str]) -> np.ndarray:
        """Embed all tasks using OpenAI's embedding model."""
        print(f"Embedding {len(tasks)} tasks using OpenAI...")
        embeddings = []
        
        for i, task in enumerate(tasks):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(tasks)}")
            
            try:
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=task
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"  ✖ Error embedding task: {e}")
                # Skip failed embeddings
                continue
            
            await asyncio.sleep(0.1)  # Rate limiting
        
        return np.array(embeddings)
    
    def cluster_tasks(self, embeddings: np.ndarray, tasks: List[str], n_clusters: int = None) -> Tuple[np.ndarray, Dict[int, List[str]]]:
        """Cluster tasks using KMeans with adaptive cluster count."""
        # Determine optimal number of clusters based on data
        if n_clusters is None:
            # Adaptive clustering: aim for clusters with at least MIN_CLUSTER_SIZE tasks
            # but no more than MAX_CLUSTER_RATIO of total tasks as clusters
            min_clusters = max(2, len(embeddings) // 20)  # At least 2, but roughly 5% granularity
            max_clusters = min(
                int(len(embeddings) * C.MAX_CLUSTER_RATIO),  # No more than 15% of tasks as clusters
                len(embeddings) // C.MIN_CLUSTER_SIZE  # Ensure minimum cluster size
            )
            n_clusters = min(max(min_clusters, 5), max_clusters)  # Between 5 and calculated max
        
        # Adjust if we have very few tasks
        actual_clusters = min(n_clusters, len(embeddings))
        print(f"Clustering {len(embeddings)} tasks into {actual_clusters} groups...")
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group tasks by cluster
        clustered_tasks = defaultdict(list)
        for task, cluster in zip(tasks, clusters):
            clustered_tasks[cluster].append(task)
        
        return clusters, dict(clustered_tasks)
    
    async def generate_cluster_names(self, clustered_tasks: Dict[int, List[str]], model: str = C.MODEL) -> Dict[int, str]:
        """Generate descriptive names for each cluster using OpenRouter."""
        print(f"Generating cluster names using {model} via OpenRouter...")
        cluster_names = {}
        
        for cluster_id, tasks in clustered_tasks.items():
            sample_tasks = tasks[:5]  # Use first 5 as representative
            tasks_text = "\n".join([f"- {task}" for task in sample_tasks])
            
            prompt = f"""Based on these money-making tasks, provide a short, descriptive category name (2-4 words):

{tasks_text}

Category name:"""
            
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Low temperature for consistent naming
                    # Optional OpenRouter-specific headers
                    extra_headers={
                        "HTTP-Referer": "https://github.com/moneybench",
                        "X-Title": "Moneybench Task Analysis"
                    }
                )
                name = response.choices[0].message.content
                if name:
                    name = name.strip().replace('"', '').replace("'", '')
                    cluster_names[cluster_id] = name
                    print(f"  Cluster {cluster_id}: {name} ({len(tasks)} tasks)")
                else:
                    cluster_names[cluster_id] = f"Cluster {cluster_id}"
            except Exception as e:
                print(f"  ✖ Error naming cluster {cluster_id}: {e}")
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        return cluster_names 