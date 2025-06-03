"""Reporter module for saving analysis results and generating reports."""

from __future__ import annotations

import json
from typing import Dict, List, Any

import constants as C


class ResultsReporter:
    """Handles saving results and generating reports."""
    
    def _sanitize_model_name(self, name: str) -> str:
        """Sanitize model name for use in filenames."""
        return name.replace("/", "_").replace(":", "-")

    def save_results(
        self,
        clustered_tasks: Dict[int, List[str]],
        cluster_names: Dict[int, str],
        raw_responses: List[Dict[str, Any]],
        timestamp: str,
        model: str = C.MODEL,
        embed_model: str = C.EMBED_MODEL
    ) -> None:
        """Save detailed results to JSON file."""
        results_data = {
            'timestamp': timestamp,
            'model': model,
            'embedding_model': embed_model,
            'num_runs': len(raw_responses),
            'total_tasks': sum(len(tasks) for tasks in clustered_tasks.values()),
            'num_clusters': len(clustered_tasks),
            'clusters': {}
        }
        
        for cluster_id, tasks in clustered_tasks.items():
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            results_data['clusters'][str(cluster_id)] = {
                'name': cluster_name,
                'num_tasks': len(tasks),
                'tasks': tasks
            }
        
        # Save full results
        full_results = {
            'summary': results_data,
            'raw_responses': raw_responses
        }
        
        s_model = self._sanitize_model_name(model)
        s_embed_model = self._sanitize_model_name(embed_model)
        filename = f'task_analysis_results_{s_model}_{s_embed_model}_{timestamp}.json'
        filepath = f'{C.ANALYSIS_DIR}/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved detailed results to {filepath}")
    
    def create_summary_report(
        self,
        clustered_tasks: Dict[int, List[str]],
        cluster_names: Dict[int, str],
        timestamp: str,
        model: str = C.MODEL,
        embed_model: str = C.EMBED_MODEL
    ) -> None:
        """Create a human-readable summary report."""
        total_tasks = sum(len(tasks) for tasks in clustered_tasks.values())
        
        report = f"""# Moneybench Task Analysis Summary
Generated: {timestamp}

## Overview
- Total tasks analyzed: {total_tasks}
- Number of clusters: {len(clustered_tasks)}
- Model used: {model}
- Embedding model: {embed_model}

## Cluster Analysis

"""
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clustered_tasks.items(), key=lambda x: len(x[1]), reverse=True)
        
        for cluster_id, tasks in sorted_clusters:
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            report += f"### {cluster_name} ({len(tasks)} tasks)\n\n"
            
            # Show all tasks in this cluster
            for i, task in enumerate(tasks, 1):
                report += f"{i}. {task}\n"
            report += "\n"
        
        # Save the report
        s_model = self._sanitize_model_name(model)
        s_embed_model = self._sanitize_model_name(embed_model)
        filename = f'summary_report_{s_model}_{s_embed_model}_{timestamp}.md'
        filepath = f'{C.ANALYSIS_DIR}/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Saved summary report to {filepath}")
    
    def print_cluster_summary(
        self,
        clustered_tasks: Dict[int, List[str]],
        cluster_names: Dict[int, str]
    ) -> None:
        """Print a summary of clusters to the terminal."""
        print(f"\n{'='*60}")
        print("CLUSTER SUMMARY:")
        print("="*60)
        
        sorted_clusters = sorted(clustered_tasks.items(), key=lambda x: len(x[1]), reverse=True)
        for cluster_id, tasks in sorted_clusters:
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            print(f"\n{cluster_name}: {len(tasks)} tasks")
            
            # Show first 3 tasks as examples
            for i, task in enumerate(tasks[:3], 1):
                print(f"  {i}. {task}")
            if len(tasks) > 3:
                print(f"  ... and {len(tasks) - 3} more") 