"""
State management for the multi-agent sentiment analysis system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class AgentState:
    """State shared between all agents in the workflow"""
    query: str = ""
    subreddit: str = ""
    post_limit: int = 10
    raw_posts: List[Dict[str, Any]] = field(default_factory=list)
    processed_posts: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_results: List[Dict[str, Any]] = field(default_factory=list)
    theme_results: List[Dict[str, Any]] = field(default_factory=list)
    final_report: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_agent: str = ""
    progress: float = 0.0
    error: str = ""
    status: str = "waiting"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
