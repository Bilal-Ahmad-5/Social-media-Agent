# Social Media Sentiment Analysis - Multi-Agent System

## Overview

This is a production-grade social media sentiment analysis system built using a multi-agent architecture. The application scrapes Reddit posts from specified subreddits, performs sentiment analysis using either HuggingFace transformer models or advanced keyword-based analysis as a fallback, extracts thematic content, and generates comprehensive reports. The system features a Streamlit dashboard for interactive visualization and uses LangGraph for workflow orchestration between different specialized agents.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Multi-Agent Architecture
The system employs a distributed agent pattern with four specialized agents:
- **RedditScraperAgent**: Handles data harvesting from Reddit using the PRAW library
- **SentimentAnalyzerAgent**: Performs sentiment analysis using transformer models or keyword-based fallback
- **ThemeTaggerAgent**: Extracts and categorizes themes from social media content
- **ReportGeneratorAgent**: Synthesizes findings into comprehensive reports

### Workflow Orchestration
Uses LangGraph (StateGraph) to manage the sequential flow between agents, ensuring proper state management and error handling. The workflow follows a linear progression: scraping → sentiment analysis → theme extraction → report generation.

### State Management
Implements a centralized state system using Python dataclasses that maintains:
- Raw and processed social media posts
- Analysis results and metrics
- Agent status and progress tracking
- Error handling and messaging

### Frontend Architecture
Built with Streamlit providing:
- Interactive dashboard with real-time updates
- Plotly visualizations for sentiment trends and theme distributions
- Configuration panels for subreddit selection and analysis parameters
- Progress monitoring and status displays

### Machine Learning Pipeline
Employs a fallback strategy for sentiment analysis:
- Primary: HuggingFace transformer models for production-grade accuracy
- Fallback: Advanced keyword-based analysis when transformers unavailable
- Supports both CPU and GPU processing with torch backend

### Data Processing Strategy
Implements asynchronous processing patterns for:
- Non-blocking Reddit API calls
- Concurrent sentiment analysis of multiple posts
- Real-time UI updates during long-running operations

## External Dependencies

### Reddit API Integration
- **PRAW (Python Reddit API Wrapper)**: For authenticated Reddit data access
- Requires Reddit API credentials (client ID and secret)
- Handles rate limiting and API authentication

### Machine Learning Services
- **HuggingFace Transformers**: Pre-trained sentiment analysis models
- **PyTorch**: Backend for neural network computations
- **NumPy**: Numerical computing for data processing

### Visualization and UI
- **Streamlit**: Web application framework for the dashboard
- **Plotly**: Interactive charts and data visualizations
- **Pandas**: Data manipulation and analysis

### Workflow Management
- **LangGraph**: Graph-based workflow orchestration
- **LangChain Core**: Runtime environment for agent coordination

### Environment Management
- **python-dotenv**: Environment variable management for API keys
- Configuration through environment variables for security

### Data Storage
Currently uses in-memory storage with dataclass state management. The architecture supports future integration with persistent storage solutions for historical analysis and data retention.