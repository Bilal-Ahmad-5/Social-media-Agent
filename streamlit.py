import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime
import json

# Import your workflow (assuming it's in the same directory)
from workflow.main import get_workflow

# Page configuration
st.set_page_config(
    page_title="🎯 Social Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        background-size: 300% 300%;
        animation: gradient 3s ease infinite;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
        backdrop-filter: blur(10px);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        height: 8px;
        border-radius: 4px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Data frame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def create_header():
    """Create animated header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Social Sentiment Analyzer</h1>
        <p>AI-Powered Reddit Sentiment & Theme Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_cards(report_data):
    """Create beautiful metric cards"""
    if not report_data or 'summary' not in report_data:
        return
    
    summary = report_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #4ECDC4; margin: 0;">📊 Total Posts</h3>
            <h2 style="color: #333; margin: 0.5rem 0 0 0;">{}</h2>
        </div>
        """.format(summary.get('total_tweets', 0)), unsafe_allow_html=True)
    
    with col2:
        sentiment_score = summary.get('sentiment_score', 0)
        color = "#4CAF50" if sentiment_score > 0 else "#FF5722" if sentiment_score < 0 else "#FF9800"
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #FF6B6B; margin: 0;">💯 Sentiment Score</h3>
            <h2 style="color: {}; margin: 0.5rem 0 0 0;">{:.2f}</h2>
        </div>
        """.format(color, sentiment_score), unsafe_allow_html=True)
    
    with col3:
        reputation = summary.get('reputation_status', 'Unknown')
        rep_colors = {
            'Excellent': '#4CAF50', 'Good': '#8BC34A', 'Neutral': '#FF9800',
            'Concerning': '#FF5722', 'Critical': '#D32F2F', 'Unknown': '#9E9E9E'
        }
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #45B7D1; margin: 0;">🏆 Reputation</h3>
            <h2 style="color: {}; margin: 0.5rem 0 0 0;">{}</h2>
        </div>
        """.format(rep_colors.get(reputation, '#9E9E9E'), reputation), unsafe_allow_html=True)
    
    with col4:
        dominant = summary.get('dominant_sentiment', 'N/A')
        dom_colors = {'POSITIVE': '#4CAF50', 'NEGATIVE': '#FF5722', 'NEUTRAL': '#FF9800'}
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #9C27B0; margin: 0;">🎭 Dominant Mood</h3>
            <h2 style="color: {}; margin: 0.5rem 0 0 0;">{}</h2>
        </div>
        """.format(dom_colors.get(dominant, '#9E9E9E'), dominant), unsafe_allow_html=True)

def create_sentiment_charts(report_data):
    """Create sentiment visualization charts"""
    if not report_data or 'sentiment_analysis' not in report_data:
        return
    
    sentiment_data = report_data['sentiment_analysis']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for sentiment distribution
        labels = list(sentiment_data['percentages'].keys())
        values = list(sentiment_data['percentages'].values())
        colors = ['#4CAF50', '#FF5722', '#FF9800']  # Green, Red, Orange
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title={
                'text': "🎭 Sentiment Distribution",
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart for sentiment counts
        sentiment_dist = sentiment_data['distribution']
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=list(sentiment_dist.keys()),
                y=list(sentiment_dist.values()),
                marker=dict(
                    color=['#4CAF50', '#FF5722', '#FF9800'],
                    line=dict(color='white', width=1)
                ),
                text=list(sentiment_dist.values()),
                textposition='auto',
                textfont=dict(size=14, color='white')
            )
        ])
        
        fig_bar.update_layout(
            title={
                'text': "📊 Sentiment Counts",
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
            yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)')
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

def create_theme_analysis(report_data):
    """Create theme analysis visualization"""
    if not report_data or 'theme_analysis' not in report_data:
        return
    
    theme_data = report_data['theme_analysis']['top_themes']
    
    if not theme_data:
        st.warning("No themes detected in the analysis.")
        return
    
    # Create horizontal bar chart for themes
    themes = list(theme_data.keys())
    counts = list(theme_data.values())
    
    # Format theme names for display
    formatted_themes = [theme.replace('_', ' ').title() for theme in themes]
    
    fig_themes = go.Figure(data=[
        go.Bar(
            x=counts,
            y=formatted_themes,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            text=counts,
            textposition='auto',
            textfont=dict(size=12, color='white')
        )
    ])
    
    fig_themes.update_layout(
        title={
            'text': "🏷️ Top Discussion Themes",
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)', title="Mentions"),
        yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        height=400
    )
    
    st.plotly_chart(fig_themes, use_container_width=True)

def display_posts_data(temp_state):
    """Display posts data in an interactive table"""
    if not temp_state.sentiment_results:
        st.warning("No posts data available.")
        return
    
    # Prepare data for display
    posts_df = []
    for i, post in enumerate(temp_state.sentiment_results, 1):
        posts_df.append({
            'Post #': i,
            'Content Preview': post.get('content', '')[:100] + "..." if len(post.get('content', '')) > 100 else post.get('content', ''),
            'Sentiment': post.get('sentiment', 'N/A'),
            'Confidence': f"{post.get('confidence', 0):.2%}",
            'Author': post.get('user', 'Unknown'),
            'Themes': ', '.join(post.get('themes', [])[:3]) if post.get('themes') else 'None'
        })
    
    df = pd.DataFrame(posts_df)
    
    # Style the dataframe
    def style_sentiment(val):
        if val == 'POSITIVE':
            return 'background-color: #4CAF50; color: white; font-weight: bold;'
        elif val == 'NEGATIVE':
            return 'background-color: #FF5722; color: white; font-weight: bold;'
        else:
            return 'background-color: #FF9800; color: white; font-weight: bold;'
    
    styled_df = df.style.applymap(style_sentiment, subset=['Sentiment'])
    
    st.markdown("### 📋 Detailed Posts Analysis")
    st.dataframe(styled_df, use_container_width=True, height=400)

def display_insights_and_recommendations(report_data):
    """Display insights and recommendations"""
    if not report_data:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💡 Key Insights")
        insights = report_data.get('insights', [])
        if insights:
            for i, insight in enumerate(insights, 1):
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.1);
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 0.5rem 0;
                    border-left: 4px solid #4ECDC4;
                    backdrop-filter: blur(10px);
                ">
                    <p style="color: white; margin: 0; font-size: 1rem;">
                        <strong>{i}.</strong> {insight}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No insights available.")
    
    with col2:
        st.markdown("### 🎯 Strategic Recommendations")
        recommendations = report_data.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.1);
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 0.5rem 0;
                    border-left: 4px solid #FF6B6B;
                    backdrop-filter: blur(10px);
                ">
                    <p style="color: white; margin: 0; font-size: 1rem;">
                        <strong>{i}.</strong> {rec}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recommendations available.")

def create_progress_animation():
    """Create animated progress indicator"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    stages = [
        ("🔍 Scraping Reddit Posts", 0.25),
        ("🧠 Analyzing Sentiment", 0.50),
        ("🏷️ Extracting Themes", 0.75),
        ("📊 Generating Report", 1.0)
    ]
    
    for stage_text, progress in stages:
        status_placeholder.markdown(f"""
        <div style="
            text-align: center;
            color: white;
            font-size: 1.2rem;
            font-weight: bold;
            margin: 1rem 0;
        ">
            {stage_text}
        </div>
        """, unsafe_allow_html=True)
        
        progress_placeholder.progress(progress)
        time.sleep(1.5)
    
    progress_placeholder.empty()
    status_placeholder.empty()

def display_model_info(report_data):
    """Display model information"""
    if not report_data or 'model_info' not in report_data:
        return
    
    model_info = report_data['model_info']
    
    st.markdown("### 🤖 AI Models Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        ">
            <h4 style="color: #4ECDC4; margin: 0;">Sentiment Analysis</h4>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                {model_info.get('sentiment_model', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        ">
            <h4 style="color: #FF6B6B; margin: 0;">Theme Extraction</h4>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                {model_info.get('theme_model', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)

def create_emotion_radar_chart(temp_state):
    """Create radar chart for emotions if available"""
    if not temp_state.sentiment_results:
        return
    
    # Aggregate emotion scores
    emotion_totals = {}
    emotion_counts = {}
    
    for result in temp_state.sentiment_results:
        emotions = result.get('emotions', {})
        for emotion, score in emotions.items():
            if emotion not in emotion_totals:
                emotion_totals[emotion] = 0
                emotion_counts[emotion] = 0
            emotion_totals[emotion] += score
            emotion_counts[emotion] += 1
    
    if not emotion_totals:
        return
    
    # Calculate averages
    emotion_averages = {
        emotion: emotion_totals[emotion] / emotion_counts[emotion]
        for emotion in emotion_totals
    }
    
    emotions = list(emotion_averages.keys())
    values = list(emotion_averages.values())
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        name='Emotion Analysis',
        line=dict(color='#4ECDC4', width=2),
        fillcolor='rgba(78, 205, 196, 0.3)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                color='white',
                gridcolor='rgba(255,255,255,0.3)'
            ),
            angularaxis=dict(color='white')
        ),
        showlegend=True,
        title={
            'text': "🎭 Emotion Analysis Radar",
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

def main():
    create_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🔧 Analysis Configuration")
        
        # Input fields
        subreddit_input = st.text_input(
            "📝 Enter Subreddit/Topic",
            placeholder="e.g., ElonMusk, Tesla, OpenAI, Apple",
            help="Enter any subreddit name or topic you want to analyze"
        )
        
        post_limit = st.slider(
            "📊 Number of Posts",
            min_value=5,
            max_value=50,
            value=15,
            help="Select how many posts to analyze"
        )
        
        analyze_button = st.button("🚀 Start Analysis", type="primary")
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Info")
        st.info("This tool analyzes Reddit sentiment using advanced AI models including RoBERTa for sentiment and BART for theme extraction.")
        
        st.markdown("### 🛠️ Features")
        st.markdown("""
        - **Real-time scraping** from Reddit
        - **Advanced sentiment** analysis
        - **Theme extraction** and categorization
        - **Comprehensive reporting** with insights
        - **Interactive visualizations**
        """)
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    
    # Main content area
    if analyze_button and subreddit_input:
        st.session_state.analysis_complete = False
        
        with st.spinner("🚀 Initializing AI Analysis Pipeline..."):
            # Create progress animation
            create_progress_animation()
            
            # Run the analysis
            try:
                workflow = get_workflow()
                
                # Run the async workflow
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    workflow.run_analysis(subreddit_input, post_limit)
                )
                loop.close()
                
                st.session_state.analysis_data = result
                st.session_state.analysis_complete = True
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please check your Reddit API credentials and try again.")
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_data:
        state = st.session_state.analysis_data
        
        # Handle both dict and object state formats
        if isinstance(state, dict):
            report_data = state.get('final_report', {})
            error = state.get('error', '')
            sentiment_results = state.get('sentiment_results', [])
            theme_results = state.get('theme_results', [])
        else:
            report_data = getattr(state, 'final_report', {})
            error = getattr(state, 'error', '')
            sentiment_results = getattr(state, 'sentiment_results', [])
            theme_results = getattr(state, 'theme_results', [])
        
        if error:
            st.error(f"❌ Error: {error}")
        else:
            # Display metrics cards
            create_metrics_cards(report_data)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Sentiment Overview", 
                "🏷️ Theme Analysis", 
                "📋 Posts Details", 
                "🎭 Emotion Analysis",
                "📄 Full Report"
            ])
            
            with tab1:
                create_sentiment_charts(report_data)
            
            with tab2:
                create_theme_analysis(report_data)
            
            with tab3:
                # Create a temporary state object for display_posts_data
                class TempState:
                    def __init__(self, sentiment_results):
                        self.sentiment_results = sentiment_results
                
                temp_state = TempState(sentiment_results)
                display_posts_data(temp_state)
            
            with tab4:
                # Create a temporary state object for emotion analysis
                temp_state = TempState(sentiment_results)
                create_emotion_radar_chart(temp_state)
            
            with tab5:
                display_insights_and_recommendations(report_data)
                
                # Display model information
                display_model_info(report_data)
                
                # Raw JSON data (expandable)
                with st.expander("🔍 Raw Analysis Data"):
                    st.json(report_data)
    
    elif not st.session_state.analysis_complete:
        # Welcome message and instructions
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            margin: 2rem 0;
        ">
            <h2 style="color: white; margin-bottom: 1rem;">🎯 Welcome to Social Sentiment Analyzer</h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 2rem;">
                Discover what people are saying about any topic on Reddit using advanced AI sentiment analysis.
            </p>
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin-top: 2rem;
            ">
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                    <h4 style="color: #4ECDC4;">🔍 Smart Scraping</h4>
                    <p style="color: white; font-size: 0.9rem;">
                        Automatically collect posts from any subreddit using Reddit API
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                    <h4 style="color: #FF6B6B;">🧠 AI Sentiment</h4>
                    <p style="color: white; font-size: 0.9rem;">
                        Advanced transformer models analyze emotional tone and sentiment
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                    <h4 style="color: #45B7D1;">🏷️ Theme Detection</h4>
                    <p style="color: white; font-size: 0.9rem;">
                        Extract key topics and themes from discussions automatically
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                    <h4 style="color: #9C27B0;">📊 Smart Reports</h4>
                    <p style="color: white; font-size: 0.9rem;">
                        Generate actionable insights and strategic recommendations
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Example topics
        st.markdown("### 💡 Try These Popular Topics:")
        
        example_cols = st.columns(5)
        examples = ["ElonMusk", "Tesla", "OpenAI", "Apple", "Bitcoin"]
        
        for i, (col, example) in enumerate(zip(example_cols, examples)):
            with col:
                if st.button(f"🎯 {example}", key=f"example_{i}"):
                    st.session_state.example_selected = example
                    st.rerun()
        
        # If example is selected, update the input
        if hasattr(st.session_state, 'example_selected'):
            st.info(f"Selected topic: {st.session_state.example_selected}")
            st.markdown("👆 **Click 'Start Analysis' in the sidebar to begin!**")

# Add footer
st.markdown("""
<div style="
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.2);
">
    <p style="color: rgba(255,255,255,0.7); margin: 0;">
        🎯 Social Sentiment Analyzer | Powered by AI & Transformers | Built with ❤️ using Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()