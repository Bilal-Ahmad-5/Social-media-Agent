"""
Social Media Sentiment Analysis System with LangGraph
A ChatGPT-style interface for analyzing social media sentiment using four AI agents
"""

import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass, field
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from agents.scraper import RedditScraperAgent
from agents.sentiment import SentimentAnalyzerAgent
from agents.theme import  ThemeTaggerAgent
from agents.report import    ReportGeneratorAgent


from workflow.state import AgentState

class SocialMediaAnalysisWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self):
        self.scraper_agent = RedditScraperAgent()
        self.sentiment_agent = SentimentAnalyzerAgent()
        self.theme_agent = ThemeTaggerAgent()
        self.report_agent = ReportGeneratorAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with four agents"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("scraper", self._scraper_node)
        workflow.add_node("sentiment_analyzer", self._sentiment_node)
        workflow.add_node("theme_tagger", self._theme_node)
        workflow.add_node("report_generator", self._report_node)
        
        # Define the workflow edges
        workflow.set_entry_point("scraper")
        workflow.add_edge("scraper", "sentiment_analyzer")
        workflow.add_edge("sentiment_analyzer", "theme_tagger")
        workflow.add_edge("theme_tagger", "report_generator")
        workflow.add_edge("report_generator", END)
        
        return workflow.compile()
    
    async def _scraper_node(self, state: AgentState) -> AgentState:
        """Twitter scraping agent node"""
        state.current_agent = "Twitter Scraper"
        state.status = "scraping"
        state.progress = 0.25
        
        try:
            # Scrape posts
            posts = await self.scraper_agent.scrape_posts(state.subreddit,state.post_limit)
            state.raw_posts = posts
            print("Scraped Raw posts")
            
        except Exception as e:
            state.error = f"Scraping failed: {str(e)}"
        return state
    
    async def _sentiment_node(self, state: AgentState) -> AgentState:
        """Sentiment analysis agent node"""
        state.current_agent = "Sentiment Analyzer"
        state.status = "analyzing_sentiment"
        state.progress = 0.50
        
        try:
            # Analyze sentiment
            results = await self.sentiment_agent.analyze_sentiment(state.raw_posts)
            state.sentiment_results = results
            print(f"Sentiment Results: {results}")
            
            # Calculate sentiment distribution
            positive = len([r for r in results if r['sentiment'] == 'POSITIVE'])
            negative = len([r for r in results if r['sentiment'] == 'NEGATIVE'])
            neutral = len([r for r in results if r['sentiment'] == 'NEUTRAL'])
        except Exception as e:
            state.error = f"Sentiment analysis failed: {str(e)}"
        return state
    
    async def _theme_node(self, state: AgentState) -> AgentState:
        """Theme tagging agent node"""
        state.current_agent = "Theme Tagger"
        state.status = "tagging_themes"
        state.progress = 0.75
        
        try:
            # Extract themes
            themes = await self.theme_agent.extract_themes(state.sentiment_results)
            state.theme_results = themes
            print(f"Theme Results: {themes}")
            # Get top themes
            theme_counts = {}
            for tweet in themes:
                for theme in tweet.get('themes', []):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            theme_text = "\n".join([f"- **{theme}**: {count} mentions" for theme, count in top_themes])
            print(f"Top Themes: {theme_text}")
        except Exception as e:
            state.error = f"Theme tagging failed: {str(e)}"
        return state
    
    async def _report_node(self, state: AgentState) -> AgentState:
        """Report generation agent node"""
        state.current_agent = "Report Generator"
        state.status = "generating_report"
        state.progress = 1.0
        
        try:
            # Generate final report
            report = await self.report_agent.generate_report(state)
            state.final_report = report
            print(f"Final Report: {report}")
            state.status = "completed"
        except Exception as e:
            state.error = f"Report generation failed: {str(e)}"
        return state
    
    async def run_analysis(self, subreddit: str, post_limit) -> AgentState:
        """Run the complete analysis workflow"""
        initial_state = AgentState(
            subreddit=subreddit,
            post_limit=post_limit,
            status="starting"
        )
        
        try:
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            print(final_state)
            return final_state
        except Exception as e:
            initial_state.error = str(e)
            initial_state.status = "error"
            return initial_state

# # Global workflow instance
# workflow_instance = None

def get_workflow():
    """Get or create the workflow instance"""
    workflow = SocialMediaAnalysisWorkflow()
    return workflow

if __name__ == "__main__":
    # Test the workflow
    async def test_workflow():
        workflow = SocialMediaAnalysisWorkflow()
        result = await workflow.run_analysis("ElonMusk",10)
        print(result)
    
    asyncio.run(test_workflow())
