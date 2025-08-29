🔍 Social Sensei - Multi-Agent System

A sophisticated multi-agent system for real-time social media sentiment analysis and thematic intelligence extraction. This production-ready solution combines transformer-based NLP with intelligent fallback mechanisms to deliver robust sentiment insights from Reddit content.

🚀 Key Features
-Multi-Agent Architecture: Four specialized agents working in concert for optimal processing
-Intelligent Fallback System: Seamless transition between transformer models and keyword analysis
-Real-Time Dashboard: Interactive Streamlit interface with live visualizations
-Theme Extraction: Advanced content categorization and trend identification
-Production Resilience: Built-in error handling and rate limiting

Core Components
-RedditScraperAgent: Handles authenticated data harvesting from Reddit using PRAW with intelligent rate limiting
-SentimentAnalyzerAgent: Dual-mode analysis with transformer models primary and keyword-based fallback--
-ThemeTaggerAgent: Content categorization and thematic intelligence extraction
-ReportGeneratorAgent: Comprehensive report synthesis with visual analytics

🚀 Quick Start

Prerequisites
-Python 3.11+
-Reddit API credentials
-(Optional) GPU for accelerated inference

Installation
-Clone the repository

bash
-git clone https://github.com/Bilal-Ahmad-5/social-sensei.git
-cd social-sensei
-Set up environment

bash
# Create virtual environment
-python -m venv venv
-source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
Configure environment variables

bash
-cp .env.example .env
# Edit .env with your API credentials:
- REDDIT_CLIENT_ID=your_client_id
- REDDIT_CLIENT_SECRET=your_client_secret
- REDDIT_USER_AGENT=your_user_agent

🎯 Usage
Start the Streamlit dashboard:

bash
-streamlit run app/dashboard.py

Configure your analysis:

-Select target subreddits
-Set post limit and time filter
-Choose analysis parameters

View real-time results:

-Sentiment distribution charts
-Theme categorization
-Interactive temporal analysis
-Exportable comprehensive reports

🔍 How It Works

Data Pipeline
-Content Acquisition: RedditScraperAgent fetches posts from specified subreddits
-Sentiment Analysis: Dual-mode analysis with automatic fallback handling
-Theme Extraction: Content categorization using advanced NLP techniques
-Report Generation: Synthesis of insights with visual analytics

Multi-Agent Orchestration
-The system uses LangGraph to manage state and workflow between specialized agents, ensuring fault tolerance and efficient resource utilization.

📊 Output Samples

The system generates:
-Sentiment distribution charts (positive/negative/neutral)
-Thematic word clouds and frequency analysis
-Temporal sentiment trends
-Exportable PDF/CSV reports
-Real-time processing metrics

🛠 Technical Stack
-Backend: Python 3.9+, LangGraph, PRAW, HuggingFace Transformers
-Frontend: Streamlit, Plotly, PandaS
-ML: PyTorch, Transformers, NumPy
-Utils: python-dotenv, dataclasses, asyncio

🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines for details.
-Fork the repository
-Create your feature branch (git checkout -b feature/AmazingFeature)
-Commit your changes (git commit -m 'Add some AmazingFeature')
-Push to the branch (git push origin feature/AmazingFeature)
-Open a Pull Request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙋‍♂️ Support

For support, please open an issue in the GitHub issue tracker or contact the maintainers.
