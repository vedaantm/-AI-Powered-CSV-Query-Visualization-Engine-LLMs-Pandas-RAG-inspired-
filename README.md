# -AI-Powered-CSV-Query-Visualization-Engine-LLMs-Pandas-RAG-inspired-
Natural language CSV query engine powered by a local LLM (Ollama + Qwen 2.5). Converts plain English questions into structured query plans, executes them on any CSV with fuzzy column matching, and auto-generates correlation-driven visualizations with LLM-written insights. Streamlit frontend. No API keys needed.
# CSV Query & Visualization Assistant

A local, privacy-first natural language interface for any CSV dataset.
Ask questions in plain English — get filtered data, aggregations,
charts, and LLM-written insights. No API keys. No cloud. Runs entirely
on your machine.

## How It Works

```
User Query (natural language)
        ↓
  LLM Query Planner (Ollama + Qwen 2.5 1.5B)
        ↓
  JSON Query Plan { filter, group_by, aggregate, sort, limit }
        ↓
  Plan Sanitizer (repairs malformed LLM output)
        ↓
  Fuzzy Column Matcher (handles typos/variations)
        ↓
  Pandas Executor → Results + Auto-Visualizations + LLM Insights
```

## Features

- **Natural Language Queries** — ask anything:
  *"Show me all employees in Engineering with salary above 80000"*
  *"What is the average salary by department, sorted descending?"*
  *"List the top 5 highest paid Data Scientists hired in 2023"*

- **Smart Query Engine** — supports 15+ filter operations
  (`equals`, `contains`, `gt`, `lt`, `between`, `month`, `year`,
  `is_null`, `startswith`, and more), group-by aggregations
  (`mean`, `median`, `sum`, `min`, `max`, `count`, `mode`, `unique`),
  AND/OR filter logic, sorting, and limiting

- **Fuzzy Column Matching** — typos and column name variations
  are automatically corrected before execution

- **Hybrid Date Detection** — combines LLM detection with regex
  and 16 date format parsers for reliable temporal queries

- **Auto-Visualization** — intelligently generates:
  - Scatter plots for correlated numeric column pairs (r > 0.4)
  - Bar charts for categorical × numeric breakdowns

- **LLM Insights** — plain English explanation of patterns and
  trends found in your query results

- **Fully Local** — powered by Ollama; your data never leaves
  your machine

## Prerequisites

Install [Ollama](https://ollama.ai) and pull the model:

```bash
ollama pull qwen2.5:1.5b
```

## Setup

```bash
git clone https://github.com/vedaantmelkari/csv-query-viz
cd csv-query-viz
pip install streamlit pandas numpy matplotlib requests
streamlit run docmain.py
```

## Tech Stack

`Python` · `Streamlit` · `Ollama` · `Qwen 2.5 1.5B` ·
`Pandas` · `NumPy` · `Matplotlib` · `difflib`

## Notes

Query accuracy depends on the local LLM. A 1.5B model handles
most queries well but may struggle with very complex multi-condition
queries — rephrase if results look unexpected.
