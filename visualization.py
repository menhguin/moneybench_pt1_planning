"""Visualization module for creating analysis plots."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import constants as C


class TaskVisualizer:
    """Creates visualizations for task analysis results."""
    
    def create_analysis_plot(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        clustered_tasks: Dict[int, List[str]],
        cluster_names: Dict[int, str],
        num_runs: int,
        model: str = C.MODEL,
        embed_model: str = C.EMBED_MODEL
    ) -> str:
        """Create the main 4-panel visualization and save it."""
        total_tasks = len(embeddings)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=C.FIGSIZE)
        fig.suptitle(
            f'Moneybench Initial Task Analysis\n'
            f'Model: {model} | Embedding: {embed_model} | '
            f'Runs: {num_runs} | Total Tasks: {total_tasks}',
            fontsize=14, y=0.98
        )
        
        # Create consistent color mapping
        unique_clusters = sorted(clustered_tasks.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
        
        # 1. Bar chart (sorted by frequency)
        self._plot_bar_chart(axes[0, 0], clustered_tasks, cluster_names, cluster_color_map, total_tasks)
        
        # 2. t-SNE visualization
        self._plot_tsne(axes[0, 1], embeddings, clusters, cluster_names, cluster_color_map)
        
        # 3. Task examples text
        self._plot_task_examples(axes[1, 0], clustered_tasks, cluster_names, total_tasks)
        
        # 4. Pie chart
        self._plot_pie_chart(axes[1, 1], clustered_tasks, cluster_names, cluster_color_map)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'task_analysis_{timestamp}.png'
        filepath = f'{C.OUTPUT_DIR}/{filename}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {filepath}")
        return timestamp
    
    def _plot_bar_chart(self, ax, clustered_tasks, cluster_names, color_map, total_tasks):
        """Create bar chart of task distribution."""
        # Sort by size
        sorted_items = sorted(clustered_tasks.items(), key=lambda x: len(x[1]), reverse=True)
        cluster_ids = [item[0] for item in sorted_items]
        counts = [len(item[1]) for item in sorted_items]
        names = [cluster_names.get(cid, f"Cluster {cid}") for cid in cluster_ids]
        colors = [color_map[cid] for cid in cluster_ids]
        
        bars = ax.bar(range(len(cluster_ids)), counts, color=colors)
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Number of Tasks')
        ax.set_title('Task Distribution by Cluster (Sorted by Frequency)')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}\n({count/total_tasks*100:.1f}%)',
                   ha='center', va='bottom', fontsize=9)
    
    def _plot_tsne(self, ax, embeddings, clusters, cluster_names, color_map):
        """Create t-SNE visualization."""
        print("Creating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot with consistent colors
        cluster_colors = [color_map[cluster] for cluster in clusters]
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  c=cluster_colors, alpha=0.6, s=30)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Task Embeddings Visualization (t-SNE)')
        
        # Legend
        unique_clusters = sorted(set(clusters))
        legend_elements = []
        for cluster_id in unique_clusters:
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color_map[cluster_id],
                          markersize=8, label=name)
            )
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_task_examples(self, ax, clustered_tasks, cluster_names, total_tasks):
        """Create text panel with task examples."""
        ax.axis('off')
        
        # Sort by size
        sorted_items = sorted(clustered_tasks.items(), key=lambda x: len(x[1]), reverse=True)
        
        text = "Top Tasks by Cluster (Sorted by Frequency):\n\n"
        for cluster_id, tasks in sorted_items:
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            percentage = len(tasks) / total_tasks * 100
            text += f"{name} ({len(tasks)} tasks, {percentage:.1f}%):\n"
            
            # Show first 2 tasks
            for task in tasks[:2]:
                if len(task) > 60:
                    text += f"  • {task[:60]}...\n"
                else:
                    text += f"  • {task}\n"
            text += "\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Example Tasks per Cluster')
    
    def _plot_pie_chart(self, ax, clustered_tasks, cluster_names, color_map):
        """Create pie chart of cluster sizes."""
        # Sort by size
        sorted_items = sorted(clustered_tasks.items(), key=lambda x: len(x[1]), reverse=True)
        cluster_ids = [item[0] for item in sorted_items]
        sizes = [len(item[1]) for item in sorted_items]
        labels = [cluster_names.get(cid, f'Cluster {cid}') for cid in cluster_ids]
        colors = [color_map[cid] for cid in cluster_ids]
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
              colors=colors, startangle=90)
        ax.set_title('Relative Cluster Sizes (Sorted by Frequency)') 