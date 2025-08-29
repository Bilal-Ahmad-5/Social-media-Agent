"""
Streamlit dashboard for multi-agent social media sentiment analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime
import json

from workflow import SentimentAnalysisWorkflow
from state import AgentState

# Page configuration
st.set_page_config(
    page_title="AI Social Sentiment Labs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin: 2rem 1rem;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
        border-radius: 15px;
        margin: 1rem 0.5rem;
    }
    
    h1 {
        color: #2D3748;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #4A5568;
        font-weight: 600;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        background: transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    .progress-container {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sentiment-positive {
        color: #10B981;
        font-weight: 600;
    }
    
    .sentiment-negative {
        color: #EF4444;
        font-weight: 600;
    }
    
    .sentiment-neutral {
        color: #6B7280;
        font-weight: 600;
    }
    
    .welcome-section {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow' not in st.session_state:
    st.session_state.workflow = SentimentAnalysisWorkflow()
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

def main():
    """Main Streamlit application"""
    
    st.markdown("""
    <h1>🧠 AI Social Sentiment Labs</h1>
    <p style="text-align: center; font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem;">
        Advanced Multi-Agent Intelligence for Social Media Analysis
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
            <h2 style="color: white; margin: 0;">⚙️ AI Control Center</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Input parameters
        subreddit = st.text_input(
            "Subreddit (without r/)", 
            value="python",
            help="Enter the subreddit name without the 'r/' prefix"
        )
        
        post_limit = st.slider(
            "Number of Posts to Analyze", 
            min_value=5, 
            max_value=50, 
            value=15,
            help="Higher numbers will take longer to process"
        )
        
        st.markdown("---")
        
        # API Configuration Info
        st.subheader("📋 API Setup")
        st.info(
            "Make sure to set your Reddit API credentials:\n"
            "- REDDIT_CLIENT_ID\n"
            "- REDDIT_CLIENT_SECRET\n\n"
            "Get these from: https://www.reddit.com/prefs/apps"
        )
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary", disabled=st.session_state.analysis_running):
            if subreddit.strip():
                st.session_state.analysis_running = True
                run_analysis(subreddit, post_limit)
            else:
                st.error("Please enter a subreddit name")
    
    # Main content area
    if st.session_state.analysis_running:
        show_analysis_progress()
    elif st.session_state.analysis_state and st.session_state.analysis_state.status == "completed":
        show_analysis_results()
    elif st.session_state.analysis_state and st.session_state.analysis_state.status == "error":
        show_error_state()
    else:
        show_welcome_screen()

def run_analysis(subreddit: str, post_limit: int):
    """Run the sentiment analysis workflow"""
    
    try:
        # Create a progress container
        progress_container = st.container()
        
        with progress_container:
            st.subheader(f"🔍 Analyzing r/{subreddit}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run the analysis
            with st.spinner("Running multi-agent analysis..."):
                # Use asyncio to run the async workflow
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(
                        st.session_state.workflow.run_analysis(subreddit, post_limit)
                    )
                    st.session_state.analysis_state = result
                finally:
                    loop.close()
        
        # Update UI state
        st.session_state.analysis_running = False
        st.rerun()
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.session_state.analysis_running = False

def show_analysis_progress():
    """Show analysis progress"""
    st.subheader("🔄 Analysis in Progress")
    
    if st.session_state.analysis_state:
        state = st.session_state.analysis_state
        
        # Progress bar
        progress = state.progress if hasattr(state, 'progress') else 0
        st.progress(progress)
        
        # Current status
        current_agent = getattr(state, 'current_agent', 'Unknown')
        status = getattr(state, 'status', 'processing')
        st.text(f"Current Agent: {current_agent}")
        st.text(f"Status: {status}")
        
        # Messages
        if hasattr(state, 'messages') and state.messages:
            st.subheader("📝 Agent Messages")
            for message in state.messages[-3:]:  # Show last 3 messages
                st.text(f"[{message.get('agent', 'Unknown')}] {message.get('message', '')}")

def show_analysis_results():
    """Show comprehensive analysis results"""
    state = st.session_state.analysis_state
    
    # Header with summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_posts = len(state.theme_results) if state.theme_results else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2rem;">{total_posts}</h3>
            <p style="margin: 0; opacity: 0.9;">Total Posts Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if state.sentiment_results:
            sentiment_counts = {}
            for item in state.sentiment_results:
                sentiment = item['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            sentiment_color = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}.get(dominant_sentiment, "⚪")
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="margin: 0; font-size: 1.5rem;">{sentiment_color} {dominant_sentiment}</h3>
                <p style="margin: 0; opacity: 0.9;">Dominant Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <h3 style="margin: 0; font-size: 1.5rem;">⚪ N/A</h3>
                <p style="margin: 0; opacity: 0.9;">Dominant Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if state.theme_results:
            theme_counts = {}
            for item in state.theme_results:
                theme = item.get('primary_theme', 'unknown')
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            dominant_theme = max(theme_counts, key=theme_counts.get)
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="margin: 0; font-size: 1.2rem;">🏷️ {dominant_theme.replace('_', ' ').title()}</h3>
                <p style="margin: 0; opacity: 0.9;">Primary Theme</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <h3 style="margin: 0; font-size: 1.2rem;">🏷️ N/A</h3>
                <p style="margin: 0; opacity: 0.9;">Primary Theme</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        subreddit = getattr(state, 'subreddit', 'unknown')
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 1.5rem;">🏠 r/{subreddit}</h3>
            <p style="margin: 0; opacity: 0.9;">Subreddit</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📝 Posts", "💭 Sentiments", "🏷️ Themes", "📈 Report"
    ])
    
    with tab1:
        show_overview_tab(state)
    
    with tab2:
        show_posts_tab(state)
    
    with tab3:
        show_sentiment_tab(state)
    
    with tab4:
        show_themes_tab(state)
    
    with tab5:
        show_report_tab(state)

def show_overview_tab(state: AgentState):
    """Show overview dashboard"""
    
    if not state.theme_results:
        st.warning("No results to display")
        return
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        sentiment_counts = {}
        for item in state.sentiment_results:
            sentiment = item['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        if sentiment_counts:
            fig_sentiment = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title="Sentiment Distribution",
                color_discrete_map={
                    'POSITIVE': '#2E8B57',
                    'NEGATIVE': '#DC143C', 
                    'NEUTRAL': '#808080'
                }
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Theme distribution bar chart
        theme_counts = {}
        for item in state.theme_results:
            theme = item.get('primary_theme', 'unknown')
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        if theme_counts:
            # Take top 10 themes
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            fig_themes = px.bar(
                x=[theme.replace('_', ' ').title() for theme, _ in sorted_themes],
                y=[count for _, count in sorted_themes],
                title="Top Themes",
                labels={'x': 'Themes', 'y': 'Number of Posts'}
            )
            fig_themes.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_themes, use_container_width=True)
    
    # Sentiment over time (using post scores as proxy)
    if state.theme_results:
        df = pd.DataFrame(state.theme_results)
        if 'score' in df.columns and 'sentiment' in df.columns:
            fig_scatter = px.scatter(
                df, 
                x='score', 
                y=df.index,  # Use index as proxy for time
                color='sentiment',
                size='confidence',
                title="Post Engagement vs Sentiment",
                labels={'x': 'Post Score', 'y': 'Post Index'},
                color_discrete_map={
                    'POSITIVE': '#2E8B57',
                    'NEGATIVE': '#DC143C',
                    'NEUTRAL': '#808080'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

def show_posts_tab(state: AgentState):
    """Show individual posts with analysis"""
    
    if not state.theme_results:
        st.warning("No posts to display")
        return
    
    st.subheader("📝 Analyzed Posts")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            options=["All"] + list(set(item['sentiment'] for item in state.theme_results))
        )
    
    with col2:
        theme_filter = st.selectbox(
            "Filter by Theme",
            options=["All"] + list(set(item.get('primary_theme', 'unknown') for item in state.theme_results))
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Score", "Confidence", "Date"]
        )
    
    # Filter and sort posts
    filtered_posts = state.theme_results
    
    if sentiment_filter != "All":
        filtered_posts = [p for p in filtered_posts if p['sentiment'] == sentiment_filter]
    
    if theme_filter != "All":
        filtered_posts = [p for p in filtered_posts if p.get('primary_theme') == theme_filter]
    
    # Sort posts
    if sort_by == "Score":
        filtered_posts = sorted(filtered_posts, key=lambda x: x.get('score', 0), reverse=True)
    elif sort_by == "Confidence":
        filtered_posts = sorted(filtered_posts, key=lambda x: x.get('confidence', 0), reverse=True)
    else:  # Date
        filtered_posts = sorted(filtered_posts, key=lambda x: x.get('created_utc', 0), reverse=True)
    
    # Display posts
    for i, post in enumerate(filtered_posts[:20]):  # Show top 20
        with st.expander(f"Post {i+1}: {post.get('title', 'No title')[:100]}..."):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Title:**", post.get('title', 'N/A'))
                st.write("**Content:**", post.get('text', 'N/A')[:500] + "..." if len(post.get('text', '')) > 500 else post.get('text', 'N/A'))
                st.write("**Author:**", post.get('author', 'N/A'))
            
            with col2:
                # Sentiment badge
                sentiment = post['sentiment']
                sentiment_color = {
                    'POSITIVE': 'green',
                    'NEGATIVE': 'red',
                    'NEUTRAL': 'gray'
                }
                st.markdown(f"**Sentiment:** :{sentiment_color.get(sentiment, 'gray')}[{sentiment}]")
                st.write(f"**Confidence:** {post.get('confidence', 0):.2f}")
                st.write(f"**Theme:** {post.get('primary_theme', 'unknown').replace('_', ' ').title()}")
                st.write(f"**Score:** {post.get('score', 0)}")
                st.write(f"**Comments:** {post.get('num_comments', 0)}")

def show_sentiment_tab(state: AgentState):
    """Show detailed sentiment analysis"""
    
    if not state.sentiment_results:
        st.warning("No sentiment data to display")
        return
    
    st.subheader("💭 Sentiment Analysis Details")
    
    # Sentiment statistics
    sentiments = [item['sentiment'] for item in state.sentiment_results]
    confidences = [item['confidence'] for item in state.sentiment_results]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_count = sentiments.count('POSITIVE')
        st.metric("Positive Posts", positive_count, f"{positive_count/len(sentiments)*100:.1f}%")
    
    with col2:
        negative_count = sentiments.count('NEGATIVE')
        st.metric("Negative Posts", negative_count, f"{negative_count/len(sentiments)*100:.1f}%")
    
    with col3:
        neutral_count = sentiments.count('NEUTRAL')
        st.metric("Neutral Posts", neutral_count, f"{neutral_count/len(sentiments)*100:.1f}%")
    
    # Confidence distribution
    fig_conf = px.histogram(
        x=confidences,
        nbins=20,
        title="Sentiment Confidence Distribution",
        labels={'x': 'Confidence Score', 'y': 'Number of Posts'}
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Model usage statistics
    models_used = [item.get('model_used', 'unknown') for item in state.sentiment_results]
    model_counts = {}
    for model in models_used:
        model_counts[model] = model_counts.get(model, 0) + 1
    
    if model_counts:
        st.subheader("🤖 Models Used")
        for model, count in model_counts.items():
            st.write(f"**{model}:** {count} posts ({count/len(models_used)*100:.1f}%)")

def show_themes_tab(state: AgentState):
    """Show detailed theme analysis"""
    
    if not state.theme_results:
        st.warning("No theme data to display")
        return
    
    st.subheader("🏷️ Theme Analysis Details")
    
    # Theme statistics
    themes = [item.get('primary_theme', 'unknown') for item in state.theme_results]
    theme_confidences = [item.get('theme_confidence', 0) for item in state.theme_results]
    
    # Theme distribution table
    theme_counts = {}
    for theme in themes:
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    theme_df = pd.DataFrame([
        {
            'Theme': theme.replace('_', ' ').title(),
            'Count': count,
            'Percentage': f"{count/len(themes)*100:.1f}%"
        }
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    st.dataframe(theme_df, use_container_width=True)
    
    # Theme confidence distribution
    if theme_confidences and any(c > 0 for c in theme_confidences):
        fig_theme_conf = px.histogram(
            x=theme_confidences,
            nbins=20,
            title="Theme Confidence Distribution",
            labels={'x': 'Confidence Score', 'y': 'Number of Posts'}
        )
        st.plotly_chart(fig_theme_conf, use_container_width=True)
    
    # Sentiment by theme
    if len(set(themes)) > 1:
        sentiment_theme_data = []
        for item in state.theme_results:
            sentiment_theme_data.append({
                'Theme': item.get('primary_theme', 'unknown').replace('_', ' ').title(),
                'Sentiment': item['sentiment']
            })
        
        sentiment_theme_df = pd.DataFrame(sentiment_theme_data)
        theme_sentiment_crosstab = pd.crosstab(sentiment_theme_df['Theme'], sentiment_theme_df['Sentiment'])
        
        fig_heatmap = px.imshow(
            theme_sentiment_crosstab.values,
            x=theme_sentiment_crosstab.columns,
            y=theme_sentiment_crosstab.index,
            title="Sentiment Distribution by Theme",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def show_report_tab(state: AgentState):
    """Show generated report"""
    
    if not state.final_report:
        st.warning("No report available")
        return
    
    report = state.final_report
    
    st.subheader("📈 Analysis Report")
    
    # Analysis Summary
    st.subheader("📋 Summary")
    summary = report.get('analysis_summary', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Posts:** {summary.get('total_posts_analyzed', 0)}")
        st.write(f"**Subreddit:** r/{summary.get('subreddit', 'unknown')}")
    with col2:
        st.write(f"**Analysis Date:** {summary.get('analysis_timestamp', 'unknown')}")
        st.write(f"**Models Used:** {', '.join(summary.get('models_used', []))}")
    
    # Sentiment Analysis Summary
    st.subheader("💭 Sentiment Summary")
    sentiment_analysis = report.get('sentiment_analysis', {})
    
    if 'distribution' in sentiment_analysis:
        for sentiment, data in sentiment_analysis['distribution'].items():
            st.write(f"**{sentiment}:** {data['count']} posts ({data['percentage']:.1f}%)")
    
    st.write(f"**Average Confidence:** {sentiment_analysis.get('average_confidence', 0):.3f}")
    st.write(f"**Dominant Sentiment:** {sentiment_analysis.get('dominant_sentiment', 'unknown')}")
    
    # Theme Analysis Summary
    st.subheader("🏷️ Theme Summary")
    theme_analysis = report.get('theme_analysis', {})
    
    if 'distribution' in theme_analysis:
        for theme, data in theme_analysis['distribution'].items():
            st.write(f"**{theme.replace('_', ' ').title()}:** {data['count']} posts ({data['percentage']:.1f}%)")
    
    st.write(f"**Average Confidence:** {theme_analysis.get('average_confidence', 0):.3f}")
    st.write(f"**Dominant Theme:** {theme_analysis.get('dominant_theme', 'unknown').replace('_', ' ').title()}")
    
    # Top Posts
    st.subheader("🔥 Top Posts by Engagement")
    top_posts = report.get('top_posts', [])
    
    for i, post in enumerate(top_posts[:5]):
        with st.expander(f"#{i+1}: {post.get('title', 'No title')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Sentiment:** {post.get('sentiment', 'unknown')}")
                st.write(f"**Theme:** {post.get('theme', 'unknown').replace('_', ' ').title()}")
            with col2:
                st.write(f"**Score:** {post.get('score', 0)}")
                st.write(f"**Engagement Ratio:** {post.get('engagement_ratio', 0):.2f}")
    
    # Insights
    st.subheader("💡 Key Insights")
    insights = report.get('insights', [])
    for insight in insights:
        st.write(f"• {insight}")
    
    # Recommendations
    st.subheader("📋 Recommendations")
    recommendations = report.get('recommendations', [])
    for recommendation in recommendations:
        st.write(f"• {recommendation}")
    
    # Export option
    st.subheader("💾 Export Report")
    if st.button("Download Report as JSON"):
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="Download JSON",
            data=report_json,
            file_name=f"sentiment_analysis_report_{summary.get('subreddit', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def show_error_state():
    """Show error state"""
    st.error("Analysis failed!")
    st.write(f"Error: {st.session_state.analysis_state.error}")
    
    if st.button("🔄 Reset and Try Again"):
        st.session_state.analysis_state = None
        st.session_state.analysis_running = False
        st.rerun()

def show_welcome_screen():
    """Show welcome screen with enhanced UI"""
    
    st.markdown("""
    <div class="welcome-section">
        <h2 style="text-align: center; color: #667eea; margin-bottom: 2rem;">
            🎉 Welcome to AI Social Sentiment Labs!
        </h2>
        
        <p style="font-size: 1.1rem; color: #4A5568; text-align: center; margin-bottom: 2rem;">
            Harness the power of advanced AI agents to unlock insights from social media conversations.
            Our multi-agent system combines cutting-edge transformer models with intelligent workflow orchestration.
        </p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
        <div class="agent-card">
            <h3>🕷️ Reddit Scraper Agent</h3>
            <p>Advanced data harvesting with async Reddit API integration. Efficiently collects posts with real-time rate limiting and error handling.</p>
        </div>
        
        <div class="agent-card">
            <h3>🧠 Sentiment Analyzer Agent</h3>
            <p>Powered by transformer models (RoBERTa + DistilRoBERTa) for production-grade sentiment analysis with emotion detection capabilities.</p>
        </div>
        
        <div class="agent-card">
            <h3>🏷️ Theme Tagger Agent</h3>
            <p>Zero-shot classification using BART-large-MNLI to identify discussion topics and themes with confidence scoring.</p>
        </div>
        
        <div class="agent-card">
            <h3>📊 Report Generator Agent</h3>
            <p>Synthesizes findings into comprehensive reports with insights, recommendations, and interactive visualizations.</p>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin: 2rem 0;">
        <h3 style="margin: 0 0 1rem 0;">🚀 Ready to Analyze?</h3>
        <p style="margin: 0; font-size: 1.1rem;">
            Configure your settings in the sidebar and click "Start Analysis" to see our AI agents work together!
        </p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0;">
        <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #059669; margin: 0.5rem 0;">⚡ Real-time Processing</h4>
            <p style="color: #374151; margin: 0;">Live Reddit data with async processing</p>
        </div>
        
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #667eea; margin: 0.5rem 0;">🤖 AI-Powered Analysis</h4>
            <p style="color: #374151; margin: 0;">Advanced transformer models</p>
        </div>
        
        <div style="background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #8B5CF6; margin: 0.5rem 0;">📈 Interactive Visualizations</h4>
            <p style="color: #374151; margin: 0;">Beautiful charts and reports</p>
        </div>
        
        <div style="background: rgba(236, 72, 153, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #EC4899; margin: 0.5rem 0;">💾 Export Capabilities</h4>
            <p style="color: #374151; margin: 0;">Download comprehensive reports</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
