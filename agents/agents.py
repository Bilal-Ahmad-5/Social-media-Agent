"""
Production-Grade Social Media Sentiment Analysis with Transformer Models
Using HuggingFace transformers for real AI-powered analysis
"""

import asyncio
import json
import re
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter
from inspect import cleandoc
import praw
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try to import transformers, fallback to basic implementation
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModel
    import torch
    import numpy as np
    HF_AVAILABLE = True
    print("HuggingFace Transformers loaded successfully!")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ HuggingFace Transformers not available, using advanced keyword-based analysis")

client_id = os.getenv("reddit_id")
client_secret = os.getenv("reddit_secret")

# Reddit Post Scraper
class RedditScraperAgent:
    """Enhanced Data Harvester with realistic social media samples"""
        
    async def scrape_posts(self, subreddit_name: str, limit: int = 50) -> List[Dict]:
        """
        Scrape posts from a subreddit using Reddit API (PRAW).
        
        Args:
            subreddit_name (str): Name of the subreddit (e.g. 'python')
            limit (int): Number of posts to fetch
        
        Returns:
            List[Dict]: List of posts with title, text, score, url, and author
        """

        # Reddit API credentials (get from https://www.reddit.com/prefs/apps)
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="RedditScraperAgent/1.0"
        )

        posts_data = []
        subreddit = reddit.subreddit(subreddit_name)

        for post in subreddit.hot(limit=limit):  # hot/new/top available
            posts_data.append({
                "title": post.title,
                "text": post.selftext,
                "score": post.score,
                "url": post.url,
                "author": str(post.author),
                "created_utc": post.created_utc
            })
            
        return posts_data

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
            print("Transformer models loaded: Twitter RoBERTa + Emotion DistilRoBERTa")
            
        except Exception as e:
            print(f"Error loading transformer models: {e}")
            self.model_loaded = False
    
    async def analyze_sentiment(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment using transformer models"""
        results = []
        
        for tweet in tweets:
            content = tweet.get('text', '')
            print(f"Content: {content}")
            
            if self.model_loaded and HF_AVAILABLE:
                try:
                    # Get sentiment from RoBERTa model
                    sentiment_result = self.sentiment_pipeline(content)[0]
                    
                    # Get emotion analysis
                    emotion_result = self.emotion_pipeline(content)[0]
                    
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
                        'content': content,
                        'user': tweet.get('user', ''),
                        'date': tweet.get('date', datetime.now().isoformat()),
                        'sentiment': mapped_sentiment,
                        'confidence': float(confidence),
                        'sentiment_scores': sentiment_scores,
                        'emotions': emotion_scores,
                        'likes': tweet.get('likes', 56),
                        'retweets': tweet.get('retweets', 9),
                        'replies': tweet.get('replies', 15),
                        'model_used': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
                    }
                    
                except Exception as e:
                    print(f"Error processing tweet with transformer: {e}")
                    result = self._fallback_analysis(tweet)
            else:
                result = self._fallback_analysis(tweet)
            
            results.append(result)
            
            # Simulate processing time
            await asyncio.sleep(0.1)
        
        return results
    
    def _fallback_analysis(self, tweet: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced keyword-based fallback analysis"""
        content = tweet.get('content', '').lower()
        
        # Enhanced sentiment lexicons
        positive_words = {
            'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'great', 'incredible', 
            'love', 'perfect', 'wonderful', 'best', 'outstanding', 'superb', 'marvelous',
            'revolutionary', 'game-changing', 'innovative', 'impressive', 'solid', 'highly',
            'recommend', 'exceeded', 'expectations', 'satisfied', 'pleased', 'thrilled'
        }
        
        negative_words = {
            'awful', 'terrible', 'horrible', 'bad', 'worst', 'hate', 'disappointed',
            'frustrated', 'annoying', 'useless', 'broken', 'failed', 'crashed', 'bugs',
            'issues', 'problems', 'regret', 'switching', 'overpriced', 'poor'
        }
        
        # Advanced pattern matching
        positive_patterns = [
            r'love\s+\w+', r'so\s+good', r'highly\s+recommend', r'exceeded\s+expectations',
            r'game[\s-]changing', r'best\s+decision', r'really\s+impressed'
        ]
        
        negative_patterns = [
            r'not\s+worth', r'waste\s+of\s+money', r'customer\s+service\s+\w*\s*terrible',
            r'crashed\s+\w+\s+times', r'so\s+frustrated', r'major\s+issues'
        ]
        
        # Count sentiment indicators
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        # Check patterns
        for pattern in positive_patterns:
            if re.search(pattern, content):
                positive_count += 2
                
        for pattern in negative_patterns:
            if re.search(pattern, content):
                negative_count += 2
        
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
            'content': content,
            'user': tweet.get('user', 'unknown'),
            'date': tweet.get('date', datetime.now().isoformat()),
            'sentiment': sentiment,
            'confidence': confidence,
            'sentiment_scores': {
                'POSITIVE': positive_count / max(positive_count + negative_count, 1),
                'NEGATIVE': negative_count / max(positive_count + negative_count, 1),
                'NEUTRAL': 0.3 if positive_count == negative_count else 0.1
            },
            'likes': tweet.get('likes', 56),
            'retweets': tweet.get('retweets', 9),
            'replies': tweet.get('replies', 15),
            'model_used': 'advanced_keyword_analysis'
        }

class ThemeTaggerAgent:
    """Advanced Theme Analysis using Transformer Models and NLP"""
    
    def __init__(self):
        self.classification_pipeline = None
        self.embedding_model = None
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
    
    async def extract_themes(self, sentiment_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract themes using transformer-based classification"""
        
        # Predefined candidate themes for classification
        candidate_themes = [
            "product_quality", "customer_service", "pricing", "user_experience", 
            "performance", "features", "reliability", "innovation", "support",
            "bugs", "updates", "design", "usability", "value", "competition",
            "recommendation", "satisfaction", "problems", "improvement", "technology"
        ]
        
        results = []
        
        for item in sentiment_results:
            content = item.get('content', '')
            
            if self.model_loaded and HF_AVAILABLE and len(content.strip()) > 10:
                try:
                    # Use zero-shot classification to identify themes
                    classification_result = self.classification_pipeline(
                        content, 
                        candidate_themes,
                        multi_label=True
                    )
                    
                    # Get themes with confidence > 0.3
                    themes = [
                        label for label, score in zip(
                            classification_result['labels'], 
                            classification_result['scores']
                        ) if score > 0.3
                    ][:5]  # Top 5 themes
                    
                    theme_scores = {
                        label: score for label, score in zip(
                            classification_result['labels'], 
                            classification_result['scores']
                        ) if score > 0.3
                    }
                    
                except Exception as e:
                    print(f"Error in transformer theme extraction: {e}")
                    themes, theme_scores = self._fallback_theme_extraction(content)
            else:
                themes, theme_scores = self._fallback_theme_extraction(content)
            
            result = {
                'content': content,
                'user': item.get('user', ''),
                'date': item.get('date', datetime.now().isoformat()),
                'sentiment': item.get('sentiment', 'NEUTRAL'),
                'confidence': item.get('confidence', 0.5),
                'themes': themes,
                'theme_scores': theme_scores,
                'model_used': 'facebook/bart-large-mnli' if self.model_loaded else 'advanced_keyword_matching'
            }
            
            results.append(result)
            await asyncio.sleep(0.05)  # Small delay for processing
        
        return results
    
    def _fallback_theme_extraction(self, content: str) -> tuple:
        """Advanced keyword-based theme extraction"""
        content_lower = content.lower()
        
        theme_keywords = {
            'product_quality': ['quality', 'reliable', 'durable', 'solid', 'well-made', 'premium'],
            'customer_service': ['support', 'service', 'help', 'staff', 'representative', 'response'],
            'pricing': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'affordable'],
            'user_experience': ['easy', 'intuitive', 'user-friendly', 'interface', 'navigation'],
            'performance': ['fast', 'slow', 'speed', 'performance', 'efficient', 'lag'],
            'features': ['feature', 'functionality', 'options', 'tools', 'capabilities'],
            'bugs': ['bug', 'error', 'crash', 'freeze', 'glitch', 'issue', 'problem'],
            'updates': ['update', 'upgrade', 'version', 'release', 'patch', 'improvement'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful', 'ugly'],
            'innovation': ['innovative', 'revolutionary', 'cutting-edge', 'advanced', 'modern']
        }
        
        detected_themes = []
        theme_scores = {}
        
        for theme, keywords in theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                detected_themes.append(theme)
                theme_scores[theme] = min(score * 0.2, 1.0)
        
        # Sort by score and return top themes
        sorted_themes = sorted(detected_themes, key=lambda x: theme_scores[x], reverse=True)
        return sorted_themes[:3], theme_scores


class ReportGeneratorAgent:
    """Advanced Insights Generator with comprehensive analytics"""
    
    async def generate_report(self, state: AgentState) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        await asyncio.sleep(1)
        
        query = state.query
        tweets = state.raw_tweets
        sentiment_results = state.sentiment_results
        theme_results = state.theme_results
        
        # Advanced sentiment analysis
        sentiments = [r.get('sentiment', 'NEUTRAL') for r in sentiment_results]
        sentiment_counts = Counter(sentiments)
        total_tweets = len(sentiments)
        
        if total_tweets == 0:
            return self._empty_report(query)
        
        # Calculate percentages and scores
        sentiment_percentages = {
            sentiment: round((count / total_tweets) * 100, 1)
            for sentiment, count in sentiment_counts.items()
        }
        
        # Advanced sentiment scoring
        positive_weight = sentiment_counts.get('POSITIVE', 0) * 1.0
        neutral_weight = sentiment_counts.get('NEUTRAL', 0) * 0.5
        negative_weight = sentiment_counts.get('NEGATIVE', 0) * -1.0
        
        sentiment_score = round((positive_weight + neutral_weight + negative_weight) / total_tweets, 2)
        
        # Theme analysis
        all_themes = []
        for result in theme_results:
            all_themes.extend(result.get('themes', []))
        
        theme_counts = Counter(all_themes)
        top_themes = dict(theme_counts.most_common(10))
        
        # Reputation status
        if sentiment_score > 0.5:
            reputation = "Excellent"
        elif sentiment_score > 0.2:
            reputation = "Good"
        elif sentiment_score > -0.2:
            reputation = "Neutral"
        elif sentiment_score > -0.5:
            reputation = "Concerning"
        else:
            reputation = "Critical"
        
        # Generate insights
        insights = self._generate_advanced_insights(
            sentiment_counts, sentiment_score, theme_counts, query
        )
        
        # Generate recommendations
        recommendations = self._generate_strategic_recommendations(
            sentiment_counts, theme_counts, reputation, query
        )
        
        # Compile comprehensive report
        report = {
            'query': query,
            'analysis_timestamp': datetime.now().isoformat(),
            'model_info': {
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest' if HF_AVAILABLE else 'advanced_keyword_analysis',
                'theme_model': 'facebook/bart-large-mnli' if HF_AVAILABLE else 'advanced_keyword_matching',
                'transformers_enabled': HF_AVAILABLE
            },
            'summary': {
                'total_tweets': total_tweets,
                'sentiment_score': sentiment_score,
                'reputation_status': reputation,
                'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get)
            },
            'sentiment_analysis': {
                'distribution': dict(sentiment_counts),
                'percentages': sentiment_percentages,
                'confidence_scores': [r.get('confidence', 0) for r in sentiment_results]
            },
            'theme_analysis': {
                'top_themes': top_themes,
                'theme_coverage': len(top_themes),
                'most_discussed': max(theme_counts, key=theme_counts.get) if theme_counts else 'None'
            },
            'insights': insights,
            'recommendations': recommendations,
            'data_quality': {
                'sample_size': total_tweets,
                'analysis_depth': 'comprehensive',
                'confidence_level': 'high' if HF_AVAILABLE else 'medium'
            }
        }
        
        return report
    
    def _generate_advanced_insights(self, sentiment_counts, sentiment_score, theme_counts, avg_engagement, query):
        """Generate intelligent insights based on analysis"""
        insights = []
        total = sum(sentiment_counts.values())
        
        if total == 0:
            return ["Insufficient data for meaningful analysis"]
        
        # Sentiment insights
        positive_ratio = sentiment_counts.get('POSITIVE', 0) / total
        negative_ratio = sentiment_counts.get('NEGATIVE', 0) / total
        
        if positive_ratio > 0.6:
            insights.append(f"{query} enjoys strong positive sentiment with {positive_ratio:.1%} favorable mentions")
        elif negative_ratio > 0.4:
            insights.append(f"{query} faces significant reputation challenges with {negative_ratio:.1%} negative sentiment")
        
        # Engagement insights
        if avg_engagement > 50:
            insights.append(f"High community engagement detected (avg {avg_engagement} interactions per post)")
        elif avg_engagement < 10:
            insights.append("Low engagement levels suggest limited brand awareness or interest")
        
        # Theme insights
        if theme_counts:
            top_theme = max(theme_counts, key=theme_counts.get)
            insights.append(f"'{top_theme.replace('_', ' ')}' is the primary discussion focus")
            
            if 'customer_service' in theme_counts and theme_counts['customer_service'] > 2:
                insights.append("Customer service experiences are a significant factor in brand perception")
            
            if 'bugs' in theme_counts or 'performance' in theme_counts:
                insights.append("Technical issues are impacting user satisfaction and require attention")
        
        return insights
    
    def _generate_strategic_recommendations(self, sentiment_counts, theme_counts, reputation, query):
        """Generate actionable strategic recommendations"""
        recommendations = []
        total = sum(sentiment_counts.values())
        
        if total == 0:
            return ["Increase brand visibility and social media presence"]
        
        negative_ratio = sentiment_counts.get('NEGATIVE', 0) / total
        
        # Reputation management
        if reputation in ["Critical", "Concerning"]:
            recommendations.append("Immediate crisis management required - address negative feedback promptly")
            recommendations.append("Implement proactive customer outreach and issue resolution protocols")
        elif reputation == "Good":
            recommendations.append("Leverage positive sentiment in marketing campaigns and testimonials")
        
        # Theme-based recommendations
        if 'customer_service' in theme_counts:
            recommendations.append("Invest in customer service training and response time improvements")
        
        if 'pricing' in theme_counts:
            recommendations.append("Conduct competitive pricing analysis and communicate value proposition clearly")
        
        if 'bugs' in theme_counts or 'performance' in theme_counts:
            recommendations.append("Prioritize product quality improvements and technical stability")
        
        if 'features' in theme_counts:
            recommendations.append("Enhance user education about existing features and gather feature requests")
        
        # General recommendations
        recommendations.append("Establish continuous social media monitoring for trend identification")
        recommendations.append("Develop content strategy targeting identified discussion themes")
        
        return recommendations
    
    def _empty_report(self, query):
        """Generate report when no data is available"""
        return {
            'query': query,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {'total_tweets': 0, 'sentiment_score': 0, 'reputation_status': 'Unknown'},
            'insights': ["No social media mentions found for analysis"],
            'recommendations': ["Increase brand visibility and social media engagement"]
        }
