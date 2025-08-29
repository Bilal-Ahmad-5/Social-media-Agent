"""
Multi-agent system for social media sentiment analysis
"""

import asyncio
import json
import re
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter
import praw
import os
from state import AgentState

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try to import transformers, fallback to basic implementation
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModel
    import torch
    import numpy as np
    HF_AVAILABLE = True
    print("✅ HuggingFace Transformers loaded successfully!")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ HuggingFace Transformers not available, using advanced keyword-based analysis")


class RedditScraperAgent:
    """Enhanced Data Harvester with Reddit API integration"""
    
    def __init__(self):
        self.client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = "SentimentAnalysisAgent/1.0"
    
    async def scrape_posts(self, state: AgentState) -> AgentState:
        """
        Scrape posts from a subreddit using Reddit API (PRAW).
        """
        try:
            state.current_agent = "Reddit Scraper"
            state.status = "scraping"
            state.progress = 0.1
            
            if not self.client_id or not self.client_secret:
                state.error = "Reddit API credentials not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
                state.status = "error"
                return state
            
            # Initialize Reddit API client
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            
            posts_data = []
            subreddit = reddit.subreddit(state.subreddit)
            
            state.progress = 0.3
            
            # Scrape posts from the subreddit
            for i, post in enumerate(subreddit.hot(limit=state.post_limit)):
                if i >= state.post_limit:
                    break
                    
                posts_data.append({
                    "id": post.id,
                    "title": post.title,
                    "text": post.selftext if post.selftext else post.title,
                    "content": f"{post.title}. {post.selftext}" if post.selftext else post.title,
                    "score": post.score,
                    "url": post.url,
                    "author": str(post.author) if post.author else "deleted",
                    "created_utc": post.created_utc,
                    "date": datetime.fromtimestamp(post.created_utc).isoformat(),
                    "num_comments": post.num_comments,
                    "upvote_ratio": post.upvote_ratio,
                    "subreddit": str(post.subreddit)
                })
                
                state.progress = 0.3 + (i / state.post_limit) * 0.4
                await asyncio.sleep(0.1)  # Rate limiting
            
            state.raw_posts = posts_data
            state.processed_posts = posts_data
            state.status = "completed"
            state.progress = 0.7
            state.messages.append({
                "agent": "Reddit Scraper",
                "message": f"Successfully scraped {len(posts_data)} posts from r/{state.subreddit}",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.error = f"Error scraping Reddit: {str(e)}"
            state.status = "error"
            state.messages.append({
                "agent": "Reddit Scraper",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state


class SentimentAnalyzerAgent:
    """Advanced Sentiment Analysis using HuggingFace Transformer Models"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        self.model_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained transformer models for sentiment analysis"""
        if not HF_AVAILABLE:
            return
            
        try:
            # Load sentiment analysis model (CardiffNLP RoBERTa)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load emotion analysis model
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            self.model_loaded = True
            print("✅ Transformer models loaded: Twitter RoBERTa + Emotion DistilRoBERTa")
            
        except Exception as e:
            print(f"⚠️ Error loading transformer models: {e}")
            self.model_loaded = False
    
    async def analyze_sentiment(self, state: AgentState) -> AgentState:
        """Analyze sentiment using transformer models"""
        try:
            state.current_agent = "Sentiment Analyzer"
            state.status = "analyzing"
            state.progress = 0.1
            
            results = []
            total_posts = len(state.processed_posts)
            
            for i, post in enumerate(state.processed_posts):
                content = post.get('content', '')
                
                if self.model_loaded and HF_AVAILABLE and len(content.strip()) > 5:
                    try:
                        # Get sentiment from RoBERTa model
                        sentiment_result = self.sentiment_pipeline(content[:512])[0]  # Truncate for model limits
                        
                        # Get emotion analysis
                        emotion_result = self.emotion_pipeline(content[:512])[0]
                        
                        # Process sentiment results
                        sentiment_scores = {item['label']: item['score'] for item in sentiment_result}
                        emotion_scores = {item['label']: item['score'] for item in emotion_result}
                        
                        # Determine primary sentiment
                        primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                        confidence = sentiment_scores[primary_sentiment]
                        
                        # Map labels to standard format
                        sentiment_mapping = {
                            'LABEL_0': 'NEGATIVE',
                            'LABEL_1': 'NEUTRAL', 
                            'LABEL_2': 'POSITIVE',
                            'NEGATIVE': 'NEGATIVE',
                            'NEUTRAL': 'NEUTRAL',
                            'POSITIVE': 'POSITIVE'
                        }
                        
                        mapped_sentiment = sentiment_mapping.get(primary_sentiment, primary_sentiment)
                        
                        result = {
                            **post,  # Include all original post data
                            'sentiment': mapped_sentiment,
                            'confidence': float(confidence),
                            'sentiment_scores': sentiment_scores,
                            'emotions': emotion_scores,
                            'primary_emotion': max(emotion_scores, key=emotion_scores.get),
                            'model_used': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
                        }
                        
                    except Exception as e:
                        print(f"Error processing post with transformer: {e}")
                        result = self._fallback_analysis(post)
                else:
                    result = self._fallback_analysis(post)
                
                results.append(result)
                state.progress = 0.1 + (i / total_posts) * 0.6
                await asyncio.sleep(0.05)  # Small delay for processing
            
            state.sentiment_results = results
            state.status = "completed"
            state.progress = 0.7
            state.messages.append({
                "agent": "Sentiment Analyzer",
                "message": f"Successfully analyzed sentiment for {len(results)} posts",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.error = f"Error in sentiment analysis: {str(e)}"
            state.status = "error"
            state.messages.append({
                "agent": "Sentiment Analyzer",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _fallback_analysis(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced keyword-based fallback analysis"""
        content = post.get('content', '').lower()
        
        # Enhanced sentiment lexicons
        positive_words = {
            'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'great', 'incredible', 
            'love', 'perfect', 'wonderful', 'best', 'outstanding', 'superb', 'marvelous',
            'revolutionary', 'game-changing', 'innovative', 'impressive', 'solid', 'highly',
            'recommend', 'exceeded', 'expectations', 'satisfied', 'pleased', 'thrilled',
            'good', 'nice', 'beautiful', 'helpful', 'useful', 'effective', 'successful'
        }
        
        negative_words = {
            'awful', 'terrible', 'horrible', 'bad', 'worst', 'hate', 'disappointed',
            'frustrated', 'annoying', 'useless', 'broken', 'failed', 'crashed', 'bugs',
            'issues', 'problems', 'regret', 'switching', 'overpriced', 'poor',
            'disgusting', 'pathetic', 'stupid', 'waste', 'scam', 'fake'
        }
        
        # Count sentiment indicators
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        # Determine sentiment with confidence
        if positive_count > negative_count:
            sentiment = 'POSITIVE'
            confidence = min(0.6 + (positive_count - negative_count) * 0.1, 0.95)
        elif negative_count > positive_count:
            sentiment = 'NEGATIVE' 
            confidence = min(0.6 + (negative_count - positive_count) * 0.1, 0.95)
        else:
            sentiment = 'NEUTRAL'
            confidence = 0.5
        
        return {
            **post,  # Include all original post data
            'sentiment': sentiment,
            'confidence': confidence,
            'sentiment_scores': {
                'POSITIVE': positive_count / max(positive_count + negative_count + 1, 1),
                'NEGATIVE': negative_count / max(positive_count + negative_count + 1, 1),
                'NEUTRAL': 1 / max(positive_count + negative_count + 1, 1)
            },
            'emotions': {'neutral': 0.8, 'other': 0.2},
            'primary_emotion': 'neutral',
            'model_used': 'advanced_keyword_analysis'
        }


class ThemeTaggerAgent:
    """Advanced Theme Analysis using Transformer Models and NLP"""
    
    def __init__(self):
        self.classification_pipeline = None
        self.model_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load transformer models for theme classification"""
        if not HF_AVAILABLE:
            return
            
        try:
            # Load zero-shot classification model
            self.classification_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            self.model_loaded = True
            print("✅ Theme extraction model loaded: BART-large MNLI")
            
        except Exception as e:
            print(f"⚠️ Error loading theme models: {e}")
            self.model_loaded = False
    
    async def extract_themes(self, state: AgentState) -> AgentState:
        """Extract themes using transformer-based classification"""
        try:
            state.current_agent = "Theme Tagger"
            state.status = "extracting"
            state.progress = 0.1
            
            # Predefined candidate themes for classification
            candidate_themes = [
                "product_quality", "customer_service", "pricing", "user_experience", 
                "performance", "features", "reliability", "innovation", "support",
                "bugs", "updates", "design", "usability", "value", "competition",
                "recommendation", "satisfaction", "problems", "improvement", "technology",
                "community", "gaming", "entertainment", "education", "news", "politics"
            ]
            
            results = []
            total_posts = len(state.sentiment_results)
            
            for i, item in enumerate(state.sentiment_results):
                content = item.get('content', '')
                
                if self.model_loaded and HF_AVAILABLE and len(content.strip()) > 10:
                    try:
                        # Use zero-shot classification to identify themes
                        classification_result = self.classification_pipeline(
                            content[:512],  # Truncate for model limits
                            candidate_themes,
                            multi_label=True
                        )
                        
                        # Get themes with confidence > 0.3
                        themes = [
                            {"theme": label, "confidence": score} 
                            for label, score in zip(
                                classification_result['labels'], 
                                classification_result['scores']
                            ) if score > 0.3
                        ]
                        
                        # Get top 3 themes
                        top_themes = themes[:3]
                        
                        result = {
                            **item,  # Include all previous data
                            'themes': top_themes,
                            'primary_theme': top_themes[0]['theme'] if top_themes else 'general',
                            'theme_confidence': top_themes[0]['confidence'] if top_themes else 0.0,
                            'theme_model_used': 'facebook/bart-large-mnli'
                        }
                        
                    except Exception as e:
                        print(f"Error processing theme with transformer: {e}")
                        result = self._fallback_theme_analysis(item)
                else:
                    result = self._fallback_theme_analysis(item)
                
                results.append(result)
                state.progress = 0.1 + (i / total_posts) * 0.6
                await asyncio.sleep(0.05)
            
            state.theme_results = results
            state.status = "completed"
            state.progress = 0.7
            state.messages.append({
                "agent": "Theme Tagger",
                "message": f"Successfully extracted themes for {len(results)} posts",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.error = f"Error in theme extraction: {str(e)}"
            state.status = "error"
            state.messages.append({
                "agent": "Theme Tagger",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _fallback_theme_analysis(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Keyword-based fallback theme analysis"""
        content = item.get('content', '').lower()
        
        # Theme keywords mapping
        theme_keywords = {
            'technology': ['tech', 'software', 'app', 'digital', 'ai', 'programming', 'code', 'developer'],
            'gaming': ['game', 'gaming', 'player', 'console', 'pc', 'mobile', 'steam'],
            'entertainment': ['movie', 'tv', 'show', 'music', 'video', 'netflix', 'youtube'],
            'education': ['learn', 'study', 'school', 'university', 'course', 'tutorial', 'education'],
            'business': ['business', 'company', 'startup', 'entrepreneur', 'market', 'sales'],
            'health': ['health', 'medical', 'doctor', 'fitness', 'wellness', 'medicine'],
            'politics': ['political', 'government', 'election', 'policy', 'vote', 'politician'],
            'community': ['community', 'people', 'social', 'group', 'forum', 'discussion'],
            'news': ['news', 'breaking', 'update', 'report', 'announcement', 'current'],
            'general': []  # Default category
        }
        
        # Count theme matches
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                theme_scores[theme] = score / len(keywords) if keywords else 0
        
        # Get primary theme
        if theme_scores:
            primary_theme = max(theme_scores, key=theme_scores.get)
            confidence = min(theme_scores[primary_theme] * 2, 0.8)  # Scale confidence
        else:
            primary_theme = 'general'
            confidence = 0.5
        
        # Create themes list
        themes = [{"theme": theme, "confidence": score} for theme, score in theme_scores.items()]
        themes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            **item,  # Include all previous data
            'themes': themes[:3],  # Top 3 themes
            'primary_theme': primary_theme,
            'theme_confidence': confidence,
            'theme_model_used': 'keyword_analysis'
        }


class ReportGeneratorAgent:
    """Generate comprehensive analysis reports"""
    
    async def generate_report(self, state: AgentState) -> AgentState:
        """Generate comprehensive analysis report"""
        try:
            state.current_agent = "Report Generator"
            state.status = "generating"
            state.progress = 0.1
            
            results = state.theme_results
            
            if not results:
                state.error = "No results to generate report from"
                state.status = "error"
                return state
            
            # Calculate sentiment distribution
            sentiment_counts = Counter(item['sentiment'] for item in results)
            total_posts = len(results)
            
            sentiment_distribution = {
                sentiment: {"count": count, "percentage": (count / total_posts) * 100}
                for sentiment, count in sentiment_counts.items()
            }
            
            state.progress = 0.3
            
            # Calculate theme distribution
            theme_counts = Counter(item['primary_theme'] for item in results)
            theme_distribution = {
                theme: {"count": count, "percentage": (count / total_posts) * 100}
                for theme, count in theme_counts.items()
            }
            
            state.progress = 0.5
            
            # Calculate average confidence scores
            avg_sentiment_confidence = sum(item['confidence'] for item in results) / total_posts
            avg_theme_confidence = sum(item.get('theme_confidence', 0) for item in results) / total_posts
            
            # Get top posts by engagement (score)
            top_posts = sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:5]
            
            state.progress = 0.7
            
            # Generate insights
            insights = self._generate_insights(results, sentiment_distribution, theme_distribution)
            
            # Create final report
            report = {
                "analysis_summary": {
                    "total_posts_analyzed": total_posts,
                    "subreddit": state.subreddit,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "models_used": list(set(item.get('model_used', 'unknown') for item in results))
                },
                "sentiment_analysis": {
                    "distribution": sentiment_distribution,
                    "average_confidence": round(avg_sentiment_confidence, 3),
                    "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get)
                },
                "theme_analysis": {
                    "distribution": theme_distribution,
                    "average_confidence": round(avg_theme_confidence, 3),
                    "dominant_theme": max(theme_counts, key=theme_counts.get)
                },
                "top_posts": [
                    {
                        "title": post.get('title', '')[:100] + '...' if len(post.get('title', '')) > 100 else post.get('title', ''),
                        "sentiment": post['sentiment'],
                        "theme": post['primary_theme'],
                        "score": post.get('score', 0),
                        "engagement_ratio": post.get('upvote_ratio', 0)
                    }
                    for post in top_posts
                ],
                "insights": insights,
                "recommendations": self._generate_recommendations(sentiment_distribution, theme_distribution)
            }
            
            state.final_report = report
            state.status = "completed"
            state.progress = 1.0
            state.messages.append({
                "agent": "Report Generator",
                "message": f"Successfully generated comprehensive report for {total_posts} posts",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.error = f"Error generating report: {str(e)}"
            state.status = "error"
            state.messages.append({
                "agent": "Report Generator",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _generate_insights(self, results: List[Dict], sentiment_dist: Dict, theme_dist: Dict) -> List[str]:
        """Generate insights from analysis results"""
        insights = []
        
        # Sentiment insights
        dominant_sentiment = max(sentiment_dist, key=lambda x: sentiment_dist[x]['count'])
        sentiment_percentage = sentiment_dist[dominant_sentiment]['percentage']
        
        if sentiment_percentage > 60:
            insights.append(f"Strong {dominant_sentiment.lower()} sentiment dominates with {sentiment_percentage:.1f}% of posts")
        elif sentiment_percentage < 40:
            insights.append("Mixed sentiment distribution indicates diverse community opinions")
        
        # Theme insights
        dominant_theme = max(theme_dist, key=lambda x: theme_dist[x]['count'])
        theme_percentage = theme_dist[dominant_theme]['percentage']
        
        if theme_percentage > 50:
            insights.append(f"Discussion heavily focused on {dominant_theme.replace('_', ' ')} ({theme_percentage:.1f}%)")
        
        # Engagement insights
        high_engagement_posts = [r for r in results if r.get('score', 0) > 100]
        if high_engagement_posts:
            avg_sentiment_high_engagement = Counter(p['sentiment'] for p in high_engagement_posts)
            most_engaging_sentiment = max(avg_sentiment_high_engagement, key=avg_sentiment_high_engagement.get)
            insights.append(f"High-engagement posts tend to be {most_engaging_sentiment.lower()}")
        
        return insights
    
    def _generate_recommendations(self, sentiment_dist: Dict, theme_dist: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Sentiment-based recommendations
        negative_percentage = sentiment_dist.get('NEGATIVE', {}).get('percentage', 0)
        positive_percentage = sentiment_dist.get('POSITIVE', {}).get('percentage', 0)
        
        if negative_percentage > 40:
            recommendations.append("Consider addressing community concerns to improve overall sentiment")
        
        if positive_percentage > 60:
            recommendations.append("Leverage positive sentiment for community engagement and growth")
        
        # Theme-based recommendations
        top_themes = sorted(theme_dist.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
        
        if top_themes:
            recommendations.append(f"Focus content strategy on top themes: {', '.join([t[0].replace('_', ' ') for t in top_themes])}")
        
        return recommendations
