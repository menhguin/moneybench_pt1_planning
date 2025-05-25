# Moneybench Initial Task Analysis

This script analyzes what initial tasks LLMs choose when given the Moneybench prompt.

## What it does

1. **Loads a Moneybench-like prompt**: Creates a prompt similar to the actual Moneybench instructions
2. **Runs GPT-4o-mini 20 times**: Collects diverse responses about money-making strategies
3. **Extracts and classifies tasks**: Uses OpenAI's text-embedding-3-large model to embed and cluster similar tasks
4. **Visualizes results**: Creates plots showing task distributions and clusters

## Setup

1. Install dependencies:
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows Command Prompt
set OPENAI_API_KEY=your-api-key-here
```

## Usage

Run the analysis:
```bash
python analyze_initial_tasks.py
```

## Output

The script generates files in organized folders:

### `output/` folder:
1. **Visualization** (`task_analysis_TIMESTAMP.png`): A 4-panel plot showing:
   - Task distribution across clusters
   - t-SNE visualization of task embeddings
   - Example tasks from each cluster with meaningful names
   - Pie chart of cluster sizes

### `analysis/` folder:
2. **Detailed results** (`task_analysis_results_TIMESTAMP.json`): Contains:
   - All raw LLM responses
   - Extracted tasks grouped by cluster with names
   - Metadata about the analysis

3. **Summary report** (`summary_report_TIMESTAMP.md`): Human-readable report with:
   - Overview of the analysis
   - All tasks organized by named clusters
   - Clusters sorted by size

## Customization

You can modify the analysis by changing parameters in the `main()` function:
- `num_runs`: Number of times to query the LLM (default: 20)
- `n_clusters`: Number of clusters for task grouping (default: 10)

## Features

- **Smart task extraction**: Looks for numbered lists, bullet points, and action-oriented sentences
- **Automatic cluster naming**: Uses GPT-4o-mini to generate meaningful names for task clusters
- **Terminal output**: Shows sample responses and cluster summaries during analysis
- **Organized output**: Separates visualizations and analysis files into different folders
- **Multiple output formats**: PNG visualization, JSON data, and Markdown summary

## Notes

- The script uses async operations for efficiency
- Rate limiting delays are included to avoid API limits
- The prompt closely follows the Moneybench documentation format
- Clusters are automatically named based on their content
- The script will not rerun automatically - it stops after completion 