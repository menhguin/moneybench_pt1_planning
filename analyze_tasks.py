#!/usr/bin/env python3
"""
Main script for analyzing initial task choices by LLMs given the Moneybench prompt.

This script:
1. Loads a prompt similar to Moneybench
2. Runs it through GPT-4o-mini multiple times
3. Uses AI to extract and summarize tasks
4. Clusters tasks using embeddings
5. Visualizes the results
"""

import asyncio
import os
import sys
from typing import List

import constants as C
from extractor import TaskExtractor
from clustering import TaskClusterer
from visualization import TaskVisualizer
from reporter import ResultsReporter


async def main():
    """Main analysis pipeline."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return 1
    
    print("Starting Moneybench Task Analysis...")
    print(f"This will collect {C.NUM_RUNS} LLM responses and analyze them.")
    print("="*60)
    
    # 1. Extract tasks from LLM responses
    extractor = TaskExtractor(api_key)
    responses = await extractor.collect_responses(C.NUM_RUNS)
    
    # Flatten all tasks
    all_tasks: List[str] = []
    for response in responses:
        all_tasks.extend(response["tasks"])
    
    print(f"\n{'='*60}")
    print(f"Collected {len(all_tasks)} total tasks from {len(responses)} runs")
    
    if not all_tasks:
        print("❌ No tasks collected. Exiting.")
        return 1
    
    # 2. Embed and cluster tasks
    clusterer = TaskClusterer(api_key)
    embeddings = await clusterer.embed_tasks(all_tasks)
    
    if len(embeddings) == 0:
        print("❌ No embeddings created. Exiting.")
        return 1
    
    clusters, clustered_tasks = clusterer.cluster_tasks(embeddings, all_tasks)
    cluster_names = await clusterer.generate_cluster_names(clustered_tasks)
    
    # 3. Show summary in terminal
    reporter = ResultsReporter()
    reporter.print_cluster_summary(clustered_tasks, cluster_names)
    
    # 4. Create visualizations
    visualizer = TaskVisualizer()
    timestamp = visualizer.create_analysis_plot(
        embeddings, clusters, clustered_tasks, cluster_names,
        num_runs=len(responses)
    )
    
    # 5. Save results
    reporter.save_results(clustered_tasks, cluster_names, responses, timestamp)
    reporter.create_summary_report(clustered_tasks, cluster_names, timestamp)
    
    print(f"\n{'='*60}")
    print("✅ Analysis complete!")
    print(f"Check the '{C.OUTPUT_DIR}' folder for visualizations")
    print(f"Check the '{C.ANALYSIS_DIR}' folder for detailed results")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1) 