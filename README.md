# Social Media Sentiment Analysis — Multi-Agent System

> A production-ready system for scraping Reddit, analyzing sentiment, extracting themes, and generating reports using a multi-agent architecture.

---

## Overview

This project scrapes Reddit posts from specified subreddits, runs sentiment analysis, extracts themes, and produces clear, shareable reports. It uses a **multi-agent** design so each part of the pipeline is handled by a focused component (agent). A Streamlit dashboard provides an interactive way to run and inspect analyses, while LangGraph handles workflow orchestration between agents.

Preferred communication style: simple, everyday language.

---

## Key Features

- Scrape Reddit posts and comments from configured subreddits.
- Perform sentiment analysis using transformer models (primary) with a keyword-based fallback.
- Extract themes and tags from posts to surface common topics.
- Generate detailed reports and visualizations (trend charts, theme distributions).
- Streamlit dashboard for configuration, live progress, and interactive results.
- LangGraph-based workflow orchestration for reliable, observable pipelines.

---

## System Architecture

### Multi-Agent Design

The system splits responsibilities across four specialized agents:

- **RedditScraperAgent** — Harvests posts and comments from Reddit using PRAW, handles pagination and rate-limiting.
- **SentimentAnalyzerAgent** — Runs sentiment analysis using HuggingFace transformer models; falls back to an advanced keyword-based method when models are unavailable.
- **ThemeTaggerAgent** — Extracts topics and tags from text and groups similar themes.
- **ReportGeneratorAgent** — Aggregates results, builds visualizations, and outputs final reports.

### Workflow Orchestration

Workflows are managed with **LangGraph** (StateGraph). The core flow is:

`scrape -> analyze sentiment -> extract themes -> generate report`

LangGraph provides state checkpoints, retries, and error handling so long-running runs are robust and observable.

### State Management

A centralized state object (implemented with Python dataclasses) keeps:

- Raw and processed posts
- Analysis outputs and aggregate metrics
- Agent progress and status
- Error logs and retry metadata

This in-memory state allows fast iteration and real-time UI updates. The architecture is ready to plug in persistent storage later.

### Frontend

Built with **Streamlit**, the dashboard includes:

- Subreddit selection and configuration panel
- Start/stop controls and run progress
- Real-time charts (Plotly) showing sentiment trends and theme distributions
- Exportable report download (CSV / PDF)

---

## Machine Learning Pipeline

- **Primary path**: transformer-based sentiment models from HuggingFace for high accuracy.
- **Fallback path**: keyword-driven sentiment scoring and heuristics for environments without GPU or when models fail.
- Supports **CPU** and **GPU** via PyTorch.
- Batch and concurrent processing of posts to improve throughput.

---

## Data Processing Strategy

- Uses asynchronous requests for non-blocking Reddit scraping.
- Concurrent sentiment analysis for faster processing of many posts.
- Chunking and batching to control memory and latency.
- Real-time updates streamed to the UI so users see progress during long runs.

---

## External Dependencies

### Reddit API Integration

- **PRAW** (Python Reddit API Wrapper) for authenticated Reddit access. Requires `client_id`, `client_secret`, and a user agent.
- Rate-limiting and backoff handling built into the scraper agent.

### Machine Learning

- **HuggingFace Transformers** for model-based sentiment analysis
- **PyTorch** as the computation backend
- **NumPy** for numeric operations

### Visualization & UI

- **Streamlit** — Dashboard and configuration UI
- **Plotly** — Interactive charts and figures
- **Pandas** — Dataframes and aggregation

### Workflow & Orchestration

- **LangGraph** — Graph-based workflow orchestration
- **LangChain Core** — Optional runtime support for agent coordination

### Environment & Config

- **python-dotenv** — Manage environment variables (Reddit keys, model config, etc.)
- All sensitive keys should come from environment variables, not hard-coded files.

---

## Data Storage

- The system uses **in-memory state** by default (dataclasses). This keeps the pipeline fast and simple for experimentation.
- The architecture supports easy integration with persistent stores later (Postgres, MongoDB, S3) for historical analysis and long-term storage.

---

## Installation & Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/Bilal-Ahmad-5/Social-media-Agent.git
cd Social-media-Agent
```

2. **Create virtual environment and install deps**

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

3. **Create a **``** file** with your Reddit credentials:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_app_user_agent
```

4. **Run the Streamlit dashboard**

```bash
streamlit run app.py
```

---

## Usage Tips

- Start with a small set of subreddits and time windows to test the pipeline.
- Use the fallback keyword analyzer when running on low-resource machines (no GPU).
- Monitor the run logs in the UI to catch rate limits or network errors early.

---

## Roadmap & Next Steps

- Add persistent storage for historical trend analysis
- Implement scheduling for recurring scraping runs
- Add authentication and multi-user support for team dashboards
- Support more social platforms (Twitter/X, Mastodon, Facebook)
- Add model monitoring and automated model-retraining pipelines

---

## Contributing

Contributions welcome. Please fork the repo, create a feature branch, add tests, and open a pull request. Include clear descriptions of your changes.

---

## License

This project is released under the **MIT License**.

---

*Simple, reliable sentiment analysis — built with modular agents and clear results.*

