"""
Workflow orchestration using LangGraph for multi-agent sentiment analysis
"""

import asyncio
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from agents.agents import (
    RedditScraperAgent, 
    SentimentAnalyzerAgent, 
    ThemeTaggerAgent, 
    ReportGeneratorAgent
)
from state import AgentState


class SentimentAnalysisWorkflow:
    """Orchestrates the multi-agent sentiment analysis workflow"""
    
    def __init__(self):
        self.reddit_scraper = RedditScraperAgent()
        self.sentiment_analyzer = SentimentAnalyzerAgent()
        self.theme_tagger = ThemeTaggerAgent()
        self.report_generator = ReportGeneratorAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("scrape_reddit", self._wrap_async(self.reddit_scraper.scrape_posts))
        workflow.add_node("analyze_sentiment", self._wrap_async(self.sentiment_analyzer.analyze_sentiment))
        workflow.add_node("extract_themes", self._wrap_async(self.theme_tagger.extract_themes))
        workflow.add_node("generate_report", self._wrap_async(self.report_generator.generate_report))
        
        # Define the workflow edges
        workflow.add_edge("scrape_reddit", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "extract_themes")
        workflow.add_edge("extract_themes", "generate_report")
        workflow.add_edge("generate_report", END)
        
        # Set entry point
        workflow.set_entry_point("scrape_reddit")
        
        return workflow.compile()
    
    def _wrap_async(self, async_func):
        """Wrap async function for LangGraph compatibility"""
        def wrapper(state: AgentState) -> AgentState:
            # Run the async function in the current event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, create a new task
                    task = asyncio.create_task(async_func(state))
                    # We can't await here, so we'll use a different approach
                    return asyncio.run_coroutine_threadsafe(async_func(state), loop).result()
                else:
                    return asyncio.run(async_func(state))
            except Exception as e:
                # If there are issues with event loops, run synchronously
                return asyncio.run(async_func(state))
        return wrapper
    
    async def run_analysis(self, subreddit: str, post_limit: int = 10) -> AgentState:
        """Run the complete sentiment analysis workflow"""
        
        # Initialize state
        initial_state = AgentState(
            subreddit=subreddit,
            post_limit=post_limit,
            status="starting"
        )
        
        try:
            # Execute the workflow
            result = await self._run_workflow_async(initial_state)
            return result
            
        except Exception as e:
            initial_state.error = f"Workflow error: {str(e)}"
            initial_state.status = "error"
            return initial_state
    
    async def _run_workflow_async(self, initial_state: AgentState) -> AgentState:
        """Run workflow asynchronously"""
        
        # Run each step of the workflow manually to maintain async compatibility
        state = initial_state
        
        # Step 1: Scrape Reddit
        state = await self.reddit_scraper.scrape_posts(state)
        if state.status == "error":
            return state
        
        # Step 2: Analyze Sentiment
        state = await self.sentiment_analyzer.analyze_sentiment(state)
        if state.status == "error":
            return state
        
        # Step 3: Extract Themes
        state = await self.theme_tagger.extract_themes(state)
        if state.status == "error":
            return state
        
        # Step 4: Generate Report
        state = await self.report_generator.generate_report(state)
        
        return state
    
    def get_workflow_status(self, state: AgentState) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "current_agent": state.current_agent,
            "status": state.status,
            "progress": state.progress,
            "error": state.error,
            "messages": state.messages[-5:],  # Last 5 messages
            "timestamp": state.timestamp
        }
    
    def get_workflow_results(self, state: AgentState) -> Dict[str, Any]:
        """Get workflow results summary"""
        return {
            "total_posts": len(state.theme_results),
            "sentiment_distribution": self._get_sentiment_distribution(state.sentiment_results),
            "theme_distribution": self._get_theme_distribution(state.theme_results),
            "has_report": bool(state.final_report),
            "status": state.status
        }
    
    def _get_sentiment_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get sentiment distribution from results"""
        if not results:
            return {}
        
        from collections import Counter
        return dict(Counter(item['sentiment'] for item in results))
    
    def _get_theme_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get theme distribution from results"""
        if not results:
            return {}
        
        from collections import Counter
        return dict(Counter(item.get('primary_theme', 'unknown') for item in results))
