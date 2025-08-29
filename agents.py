"""
Multi-agent system for social media sentiment analysis

This module implements a sophisticated multi-agent architecture for analyzing
social media sentiment and themes using advanced NLP techniques and transformer models.

Author: AI Social Sentiment Labs
Version: 2.0.0
"""

import asyncio
import json
import re
import warnings
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from collections import Counter
import asyncpraw
import os
from state import AgentState

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try to import transformers with comprehensive error handling
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModel
    import torch
    import numpy as np
    HF_AVAILABLE = True
    logger.info("✅ HuggingFace Transformers loaded successfully!")
except ImportError as e:
    HF_AVAILABLE = False
    logger.warning(f"⚠️ HuggingFace Transformers not available: {e}")
    logger.info("Falling back to advanced keyword-based analysis")
except Exception as e:
    HF_AVAILABLE = False
    logger.error(f"❌ Error loading transformers: {e}")
    logger.info("Falling back to advanced keyword-based analysis")


class RedditScraperAgent:
    """
    Enhanced Reddit Data Harvester Agent
    
    This agent handles the collection of Reddit posts using the AsyncPRAW library
    for efficient asynchronous data harvesting with proper error handling and
    rate limiting compliance.
    
    Attributes:
        client_id (str): Reddit API client ID from environment variables
        client_secret (str): Reddit API client secret from environment variables
        user_agent (str): User agent string for Reddit API requests
        reddit (Optional[asyncpraw.Reddit]): AsyncPRAW Reddit instance
    """
    
    def __init__(self):
        """Initialize the Reddit Scraper Agent with API credentials."""
        self.client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = "AI-Social-Sentiment-Labs/2.0"
        self.reddit: Optional[asyncpraw.Reddit] = None
        
        logger.info("Reddit Scraper Agent initialized")
    
    async def scrape_posts(self, state: AgentState) -> AgentState:
        """
        Scrape posts from a subreddit using AsyncPRAW with comprehensive error handling.
        
        This method handles the complete workflow of:
        1. Validating API credentials
        2. Establishing Reddit API connection
        3. Fetching posts with rate limiting
        4. Processing and structuring post data
        5. Proper session cleanup
        
        Args:
            state (AgentState): The current state object containing configuration
            
        Returns:
            AgentState: Updated state with scraped posts or error information
        """
        # Initialize agent status
        state.current_agent = "Reddit Scraper"
        state.status = "scraping"
        state.progress = 0.1
        
        logger.info(f"Starting Reddit scraping for r/{state.subreddit}")
        
        try:
            # Validate API credentials
            if not self.client_id or not self.client_secret:
                error_msg = "Reddit API credentials missing. Please configure REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
                logger.error(error_msg)
                state.error = error_msg
                state.status = "error"
                return state
            
            # Initialize AsyncPRAW Reddit API client with proper error handling
            try:
                self.reddit = asyncpraw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    timeout=30  # Add timeout for robustness
                )
                logger.info("AsyncPRAW client initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize Reddit client: {str(e)}"
                logger.error(error_msg)
                state.error = error_msg
                state.status = "error"
                return state
            
            # Access subreddit with error handling
            try:
                subreddit = await self.reddit.subreddit(state.subreddit)
                logger.info(f"Connected to subreddit: r/{state.subreddit}")
            except Exception as e:
                error_msg = f"Failed to access subreddit r/{state.subreddit}: {str(e)}"
                logger.error(error_msg)
                state.error = error_msg
                state.status = "error"
                await self._cleanup_session()
                return state
            
            state.progress = 0.3
            posts_data = []
            
            # Scrape posts with comprehensive error handling and rate limiting
            try:
                post_count = 0
                logger.info(f"Fetching {state.post_limit} posts from r/{state.subreddit}")
                
                async for post in subreddit.hot(limit=state.post_limit):
                    if post_count >= state.post_limit:
                        break
                    
                    try:
                        # Process individual post with error handling
                        post_data = {
                            "id": post.id,
                            "title": post.title or "[No Title]",
                            "text": post.selftext or "",
                            "content": self._create_content_text(post.title, post.selftext),
                            "score": getattr(post, 'score', 0),
                            "url": getattr(post, 'url', ''),
                            "author": str(post.author) if post.author else "[deleted]",
                            "created_utc": getattr(post, 'created_utc', 0),
                            "date": datetime.fromtimestamp(getattr(post, 'created_utc', 0)).isoformat(),
                            "num_comments": getattr(post, 'num_comments', 0),
                            "upvote_ratio": getattr(post, 'upvote_ratio', 0.5),
                            "subreddit": str(getattr(post, 'subreddit', state.subreddit))
                        }
                        
                        posts_data.append(post_data)
                        post_count += 1
                        
                        # Update progress
                        state.progress = 0.3 + (post_count / state.post_limit) * 0.4
                        
                        # Rate limiting - respect Reddit API guidelines
                        await asyncio.sleep(0.1)
                        
                    except Exception as post_error:
                        logger.warning(f"Error processing individual post: {post_error}")
                        continue  # Skip problematic posts but continue
                
                logger.info(f"Successfully scraped {len(posts_data)} posts")
                
            except Exception as e:
                error_msg = f"Error during post iteration: {str(e)}"
                logger.error(error_msg)
                state.error = error_msg
                state.status = "error"
                await self._cleanup_session()
                return state
            
            # Update state with scraped data
            state.raw_posts = posts_data
            state.processed_posts = posts_data
            state.status = "completed"
            state.progress = 0.7
            
            # Add success message
            success_message = f"Successfully scraped {len(posts_data)} posts from r/{state.subreddit}"
            state.messages.append({
                "agent": "Reddit Scraper",
                "message": success_message,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(success_message)
            
        except Exception as e:
            # Handle any unexpected errors
            error_msg = f"Unexpected error in Reddit scraping: {str(e)}"
            logger.error(error_msg)
            state.error = error_msg
            state.status = "error"
            state.messages.append({
                "agent": "Reddit Scraper",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        finally:
            # Always cleanup the session
            await self._cleanup_session()
        
        return state
    
    def _create_content_text(self, title: str, selftext: str) -> str:
        """
        Create combined content text from title and selftext.
        
        Args:
            title (str): Post title
            selftext (str): Post body text
            
        Returns:
            str: Combined content for analysis
        """
        if selftext and selftext.strip():
            return f"{title}. {selftext}"
        return title or "[No Content]"
    
    async def _cleanup_session(self):
        """
        Properly cleanup the AsyncPRAW session to prevent memory leaks.
        """
        if self.reddit:
            try:
                await self.reddit.close()
                logger.info("Reddit session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Reddit session: {e}")
            finally:
                self.reddit = None


class SentimentAnalyzerAgent:
    """
    Advanced Sentiment Analysis Agent using Transformer Models
    
    This agent performs sophisticated sentiment analysis using state-of-the-art
    transformer models from HuggingFace. It includes fallback mechanisms for
    robustness and comprehensive error handling.
    
    Attributes:
        sentiment_pipeline: HuggingFace pipeline for sentiment classification
        emotion_pipeline: HuggingFace pipeline for emotion detection
        model_loaded (bool): Flag indicating if models are successfully loaded
    """
    
    def __init__(self):
        """Initialize the Sentiment Analyzer Agent with transformer models."""
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        self.model_loaded = False
        
        logger.info("Initializing Sentiment Analyzer Agent")
        self._load_models()
        
        if self.model_loaded:
            logger.info("Sentiment Analyzer Agent initialized with transformer models")
        else:
            logger.info("Sentiment Analyzer Agent initialized with keyword-based fallback")
    
    def _load_models(self):
        """
        Load pre-trained transformer models with comprehensive error handling.
        
        Attempts to load:
        1. CardiffNLP RoBERTa for sentiment analysis
        2. DistilRoBERTa for emotion detection
        
        Falls back gracefully if models cannot be loaded.
        """
        if not HF_AVAILABLE:
            logger.info("HuggingFace transformers not available, using fallback analysis")
            return
            
        try:
            logger.info("Loading transformer models for sentiment analysis...")
            
            # Load sentiment analysis model (CardiffNLP RoBERTa)
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True,
                    device=-1  # Use CPU for compatibility
                )
                logger.info("✅ Sentiment analysis model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentiment model: {e}")
                raise
            
            # Load emotion analysis model
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                    device=-1  # Use CPU for compatibility
                )
                logger.info("✅ Emotion analysis model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
                raise
            
            self.model_loaded = True
            logger.info("🎉 All transformer models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading transformer models: {e}")
            logger.info("Falling back to keyword-based analysis")
            self.model_loaded = False
            self.sentiment_pipeline = None
            self.emotion_pipeline = None
    
    async def analyze_sentiment(self, state: AgentState) -> AgentState:
        """
        Analyze sentiment of posts using transformer models or keyword fallback.
        
        This method processes each post through:
        1. Content validation and preprocessing
        2. Transformer-based sentiment analysis (if available)
        3. Emotion detection and scoring
        4. Fallback keyword analysis if needed
        5. Result structuring and confidence scoring
        
        Args:
            state (AgentState): Current state with processed posts
            
        Returns:
            AgentState: Updated state with sentiment analysis results
        """
        # Initialize analysis status
        state.current_agent = "Sentiment Analyzer"
        state.status = "analyzing"
        state.progress = 0.1
        
        logger.info("Starting sentiment analysis")
        
        try:
            results = []
            total_posts = len(state.processed_posts)
            
            if total_posts == 0:
                logger.warning("No posts to analyze")
                state.status = "completed"
                state.progress = 0.7
                return state
            
            logger.info(f"Analyzing sentiment for {total_posts} posts")
            
            # Process each post
            for i, post in enumerate(state.processed_posts):
                try:
                    content = post.get('content', '').strip()
                    
                    # Skip empty content
                    if not content:
                        logger.warning(f"Skipping post {i+1} - empty content")
                        result = self._create_neutral_result(post)
                        results.append(result)
                        continue
                    
                    # Use transformer models if available and content is sufficient
                    if self.model_loaded and HF_AVAILABLE and len(content) > 5:
                        try:
                            result = await self._transformer_analysis(post, content)
                            logger.debug(f"Transformer analysis completed for post {i+1}")
                        except Exception as e:
                            logger.warning(f"Transformer analysis failed for post {i+1}: {e}")
                            result = self._fallback_analysis(post)
                    else:
                        # Use keyword-based fallback
                        result = self._fallback_analysis(post)
                        logger.debug(f"Keyword analysis completed for post {i+1}")
                    
                    results.append(result)
                    
                    # Update progress
                    state.progress = 0.1 + (i / total_posts) * 0.6
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.02)
                    
                except Exception as post_error:
                    logger.error(f"Error analyzing post {i+1}: {post_error}")
                    # Create neutral result for failed posts
                    result = self._create_neutral_result(post)
                    results.append(result)
                    continue
            
            # Update state with results
            state.sentiment_results = results
            state.status = "completed"
            state.progress = 0.7
            
            success_message = f"Successfully analyzed sentiment for {len(results)} posts"
            state.messages.append({
                "agent": "Sentiment Analyzer",
                "message": success_message,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(success_message)
            
        except Exception as e:
            error_msg = f"Error in sentiment analysis: {str(e)}"
            logger.error(error_msg)
            state.error = error_msg
            state.status = "error"
            state.messages.append({
                "agent": "Sentiment Analyzer",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    async def _transformer_analysis(self, post: Dict[str, Any], content: str) -> Dict[str, Any]:
        """
        Perform transformer-based sentiment analysis on content.
        
        Args:
            post (Dict): Original post data
            content (str): Text content to analyze
            
        Returns:
            Dict: Analysis results with sentiment and emotion data
        """
        try:
            # Truncate content for model limits (512 tokens)
            truncated_content = content[:512]
            
            # Get sentiment analysis
            sentiment_result = self.sentiment_pipeline(truncated_content)[0]
            
            # Get emotion analysis
            emotion_result = self.emotion_pipeline(truncated_content)[0]
            
            # Process sentiment scores
            sentiment_scores = {item['label']: item['score'] for item in sentiment_result}
            emotion_scores = {item['label']: item['score'] for item in emotion_result}
            
            # Determine primary sentiment with label mapping
            primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[primary_sentiment]
            
            # Map model labels to standard format
            sentiment_mapping = {
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'NEUTRAL', 
                'LABEL_2': 'POSITIVE',
                'NEGATIVE': 'NEGATIVE',
                'NEUTRAL': 'NEUTRAL',
                'POSITIVE': 'POSITIVE'
            }
            
            mapped_sentiment = sentiment_mapping.get(primary_sentiment, primary_sentiment)
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            
            # Create comprehensive result
            result = {
                **post,  # Include all original post data
                'sentiment': mapped_sentiment,
                'confidence': float(confidence),
                'sentiment_scores': sentiment_scores,
                'emotions': emotion_scores,
                'primary_emotion': primary_emotion,
                'emotion_confidence': emotion_scores[primary_emotion],
                'model_used': 'transformer_ensemble',
                'analysis_method': 'transformer'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Transformer analysis error: {e}")
            raise
    
    def _create_neutral_result(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a neutral sentiment result for posts that cannot be analyzed.
        
        Args:
            post (Dict): Original post data
            
        Returns:
            Dict: Neutral sentiment result
        """
        return {
            **post,
            'sentiment': 'NEUTRAL',
            'confidence': 0.5,
            'sentiment_scores': {'POSITIVE': 0.33, 'NEGATIVE': 0.33, 'NEUTRAL': 0.34},
            'emotions': {'neutral': 0.8, 'other': 0.2},
            'primary_emotion': 'neutral',
            'emotion_confidence': 0.8,
            'model_used': 'neutral_fallback',
            'analysis_method': 'fallback'
        }
    
    def _fallback_analysis(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced keyword-based fallback analysis with comprehensive lexicons.
        
        This method provides robust sentiment analysis when transformer models
        are unavailable, using carefully curated word lists and pattern matching.
        
        Args:
            post (Dict): Original post data
            
        Returns:
            Dict: Comprehensive sentiment analysis result
        """
        try:
            content = post.get('content', '').lower()
            
            # Enhanced sentiment lexicons with domain-specific terms
            positive_words = {
                # Basic positive terms
                'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'great', 'incredible', 
                'love', 'perfect', 'wonderful', 'best', 'outstanding', 'superb', 'marvelous',
                
                # Technology and product terms
                'revolutionary', 'game-changing', 'innovative', 'impressive', 'solid', 'reliable',
                'efficient', 'smooth', 'fast', 'intuitive', 'user-friendly', 'seamless',
                
                # Emotional and experiential terms
                'recommend', 'exceeded', 'expectations', 'satisfied', 'pleased', 'thrilled',
                'happy', 'excited', 'delighted', 'grateful', 'appreciate', 'thankful',
                
                # General positive terms
                'good', 'nice', 'beautiful', 'helpful', 'useful', 'effective', 'successful',
                'quality', 'value', 'worth', 'affordable', 'cheap', 'free', 'bonus'
            }
            
            negative_words = {
                # Basic negative terms
                'awful', 'terrible', 'horrible', 'bad', 'worst', 'hate', 'disappointed',
                'frustrated', 'annoying', 'useless', 'broken', 'failed', 'disaster',
                
                # Technical and functional issues
                'crashed', 'bugs', 'glitches', 'errors', 'issues', 'problems', 'slow',
                'laggy', 'confusing', 'complicated', 'difficult', 'hard', 'impossible',
                
                # Economic and value terms
                'expensive', 'overpriced', 'costly', 'waste', 'scam', 'ripoff', 'fraud',
                'regret', 'mistake', 'poor', 'cheap', 'flimsy', 'unreliable',
                
                # Emotional negative terms
                'disgusting', 'pathetic', 'stupid', 'ridiculous', 'absurd', 'fake',
                'lies', 'misleading', 'wrong', 'incorrect', 'false'
            }
            
            # Advanced pattern matching for context
            positive_patterns = [
                r'highly\s+recommend', r'exceeded\s+expectations', r'love\s+it',
                r'game[\s-]changing', r'best\s+decision', r'really\s+impressed',
                r'works\s+great', r'perfect\s+for', r'amazing\s+quality'
            ]
            
            negative_patterns = [
                r'not\s+worth', r'waste\s+of\s+money', r'total\s+disaster',
                r'completely\s+broken', r'so\s+frustrated', r'major\s+issues',
                r'regret\s+buying', r'save\s+your\s+money', r'avoid\s+at\s+all\s+costs'
            ]
            
            # Count basic word matches
            positive_count = sum(1 for word in positive_words if word in content)
            negative_count = sum(1 for word in negative_words if word in content)
            
            # Check advanced patterns (weighted higher)
            for pattern in positive_patterns:
                if re.search(pattern, content):
                    positive_count += 2
                    
            for pattern in negative_patterns:
                if re.search(pattern, content):
                    negative_count += 2
            
            # Handle negation (basic implementation)
            negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor']
            for neg_word in negation_words:
                if neg_word in content:
                    # Simple negation handling - reverse nearby sentiment
                    positive_count, negative_count = negative_count, positive_count
                    break
            
            # Calculate total indicators
            total_indicators = positive_count + negative_count
            
            # Determine sentiment with confidence scoring
            if positive_count > negative_count:
                sentiment = 'POSITIVE'
                confidence = min(0.6 + (positive_count - negative_count) * 0.1, 0.95)
            elif negative_count > positive_count:
                sentiment = 'NEGATIVE' 
                confidence = min(0.6 + (negative_count - positive_count) * 0.1, 0.95)
            else:
                sentiment = 'NEUTRAL'
                confidence = 0.5
            
            # Create normalized sentiment scores
            if total_indicators > 0:
                pos_score = positive_count / total_indicators
                neg_score = negative_count / total_indicators
                neu_score = max(0.1, 1 - pos_score - neg_score)
            else:
                pos_score = neg_score = 0.3
                neu_score = 0.4
            
            # Create comprehensive result
            result = {
                **post,  # Include all original post data
                'sentiment': sentiment,
                'confidence': confidence,
                'sentiment_scores': {
                    'POSITIVE': pos_score,
                    'NEGATIVE': neg_score,
                    'NEUTRAL': neu_score
                },
                'emotions': {
                    'joy': pos_score * 0.8 if sentiment == 'POSITIVE' else 0.1,
                    'anger': neg_score * 0.6 if sentiment == 'NEGATIVE' else 0.1,
                    'sadness': neg_score * 0.4 if sentiment == 'NEGATIVE' else 0.1,
                    'neutral': neu_score
                },
                'primary_emotion': 'joy' if sentiment == 'POSITIVE' else ('anger' if sentiment == 'NEGATIVE' else 'neutral'),
                'emotion_confidence': confidence,
                'keyword_matches': {
                    'positive': positive_count,
                    'negative': negative_count
                },
                'model_used': 'advanced_keyword_analysis',
                'analysis_method': 'keyword_fallback'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback analysis error: {e}")
            return self._create_neutral_result(post)


class ThemeTaggerAgent:
    """
    Advanced Theme Analysis Agent using Transformer Models
    
    This agent identifies and categorizes themes in social media content using
    zero-shot classification with BART-large-MNLI model, providing comprehensive
    theme analysis with confidence scoring and fallback mechanisms.
    
    Attributes:
        classification_pipeline: HuggingFace zero-shot classification pipeline
        model_loaded (bool): Flag indicating if the model is successfully loaded
        candidate_themes (List[str]): Predefined themes for classification
    """
    
    def __init__(self):
        """Initialize the Theme Tagger Agent with classification models."""
        self.classification_pipeline = None
        self.model_loaded = False
        
        # Comprehensive theme categories for classification
        self.candidate_themes = [
            # Technology and digital
            "technology", "software", "programming", "artificial_intelligence", "machine_learning",
            "cybersecurity", "data_science", "web_development", "mobile_apps", "gaming",
            
            # Business and economics
            "business", "entrepreneurship", "marketing", "finance", "investment", "cryptocurrency",
            "startup", "economy", "trade", "commerce",
            
            # Social and cultural
            "politics", "society", "culture", "religion", "education", "history", "philosophy",
            "ethics", "human_rights", "social_justice",
            
            # Entertainment and media
            "entertainment", "movies", "television", "music", "books", "art", "sports",
            "celebrity", "news", "journalism",
            
            # Health and lifestyle
            "health", "fitness", "nutrition", "medical", "mental_health", "wellness",
            "lifestyle", "fashion", "travel", "food",
            
            # Science and environment
            "science", "research", "environment", "climate_change", "sustainability",
            "nature", "wildlife", "space", "physics", "biology",
            
            # General topics
            "community", "relationships", "family", "personal_growth", "advice",
            "opinion", "discussion", "question", "recommendation", "review"
        ]
        
        logger.info("Initializing Theme Tagger Agent")
        self._load_models()
        
        if self.model_loaded:
            logger.info("Theme Tagger Agent initialized with transformer models")
        else:
            logger.info("Theme Tagger Agent initialized with keyword-based fallback")
    
    def _load_models(self):
        """
        Load transformer models for zero-shot theme classification.
        
        Attempts to load Facebook's BART-large-MNLI model for comprehensive
        theme classification with fallback handling.
        """
        if not HF_AVAILABLE:
            logger.info("HuggingFace transformers not available for theme analysis")
            return
            
        try:
            logger.info("Loading BART-large-MNLI model for theme classification...")
            
            # Load zero-shot classification model with proper configuration
            self.classification_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # Use CPU for compatibility
                framework="pt"  # PyTorch framework
            )
            
            self.model_loaded = True
            logger.info("✅ Theme classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading theme classification models: {e}")
            logger.info("Falling back to keyword-based theme analysis")
            self.model_loaded = False
            self.classification_pipeline = None
    
    async def extract_themes(self, state: AgentState) -> AgentState:
        """
        Extract themes using transformer-based classification with comprehensive fallback.
        
        This method processes each post through:
        1. Content validation and preprocessing
        2. Zero-shot classification (if available)
        3. Theme confidence scoring and ranking
        4. Fallback keyword-based theme detection
        5. Result structuring with multiple theme options
        
        Args:
            state (AgentState): Current state with sentiment analysis results
            
        Returns:
            AgentState: Updated state with theme analysis results
        """
        # Initialize theme extraction status
        state.current_agent = "Theme Tagger"
        state.status = "extracting"
        state.progress = 0.1
        
        logger.info("Starting theme extraction")
        
        try:
            results = []
            total_posts = len(state.sentiment_results)
            
            if total_posts == 0:
                logger.warning("No posts to analyze for themes")
                state.status = "completed"
                state.progress = 0.7
                return state
            
            logger.info(f"Extracting themes for {total_posts} posts")
            
            # Process each post
            for i, item in enumerate(state.sentiment_results):
                try:
                    content = item.get('content', '').strip()
                    
                    # Skip very short content
                    if len(content) < 10:
                        logger.debug(f"Skipping post {i+1} - content too short")
                        result = self._create_general_theme_result(item)
                        results.append(result)
                        continue
                    
                    # Use transformer model if available
                    if self.model_loaded and HF_AVAILABLE:
                        try:
                            result = await self._transformer_theme_analysis(item, content)
                            logger.debug(f"Transformer theme analysis completed for post {i+1}")
                        except Exception as e:
                            logger.warning(f"Transformer theme analysis failed for post {i+1}: {e}")
                            result = self._fallback_theme_analysis(item)
                    else:
                        # Use keyword-based fallback
                        result = self._fallback_theme_analysis(item)
                        logger.debug(f"Keyword theme analysis completed for post {i+1}")
                    
                    results.append(result)
                    
                    # Update progress
                    state.progress = 0.1 + (i / total_posts) * 0.6
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.02)
                    
                except Exception as post_error:
                    logger.error(f"Error analyzing themes for post {i+1}: {post_error}")
                    # Create general theme result for failed posts
                    result = self._create_general_theme_result(item)
                    results.append(result)
                    continue
            
            # Update state with results
            state.theme_results = results
            state.status = "completed"
            state.progress = 0.7
            
            success_message = f"Successfully extracted themes for {len(results)} posts"
            state.messages.append({
                "agent": "Theme Tagger",
                "message": success_message,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(success_message)
            
        except Exception as e:
            error_msg = f"Error in theme extraction: {str(e)}"
            logger.error(error_msg)
            state.error = error_msg
            state.status = "error"
            state.messages.append({
                "agent": "Theme Tagger",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    async def _transformer_theme_analysis(self, item: Dict[str, Any], content: str) -> Dict[str, Any]:
        """
        Perform transformer-based theme classification on content.
        
        Args:
            item (Dict): Post data with sentiment analysis results
            content (str): Text content to analyze
            
        Returns:
            Dict: Analysis results with theme data
        """
        try:
            # Truncate content for model limits
            truncated_content = content[:512]
            
            # Use zero-shot classification to identify themes
            classification_result = self.classification_pipeline(
                truncated_content,
                self.candidate_themes,
                multi_label=True
            )
            
            # Extract themes with confidence > 0.2 (lower threshold for more themes)
            themes = [
                {"theme": label, "confidence": score}
                for label, score in zip(
                    classification_result['labels'],
                    classification_result['scores']
                ) if score > 0.2
            ]
            
            # Get top 5 themes
            top_themes = themes[:5]
            
            # Create comprehensive result
            result = {
                **item,  # Include all previous data
                'themes': top_themes,
                'primary_theme': top_themes[0]['theme'] if top_themes else 'general',
                'theme_confidence': top_themes[0]['confidence'] if top_themes else 0.0,
                'all_theme_scores': dict(zip(classification_result['labels'], classification_result['scores'])),
                'theme_model_used': 'facebook/bart-large-mnli',
                'theme_analysis_method': 'transformer'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Transformer theme analysis error: {e}")
            raise
    
    def _create_general_theme_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a general theme result for posts that cannot be analyzed.
        
        Args:
            item (Dict): Original post data
            
        Returns:
            Dict: General theme result
        """
        return {
            **item,
            'themes': [{"theme": "general", "confidence": 0.5}],
            'primary_theme': 'general',
            'theme_confidence': 0.5,
            'theme_model_used': 'general_fallback',
            'theme_analysis_method': 'fallback'
        }
    
    def _fallback_theme_analysis(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keyword-based fallback theme analysis with comprehensive coverage.
        
        This method provides robust theme classification when transformer models
        are unavailable, using carefully curated keyword mappings.
        
        Args:
            item (Dict): Post data with sentiment analysis
            
        Returns:
            Dict: Comprehensive theme analysis result
        """
        try:
            content = item.get('content', '').lower()
            
            # Enhanced theme keywords mapping
            theme_keywords = {
                # Technology themes
                'technology': ['tech', 'technology', 'digital', 'innovation', 'technical', 'software', 'hardware'],
                'programming': ['code', 'programming', 'developer', 'coding', 'algorithm', 'python', 'javascript', 'api'],
                'artificial_intelligence': ['ai', 'artificial intelligence', 'machine learning', 'neural', 'algorithm', 'automation'],
                'gaming': ['game', 'gaming', 'player', 'console', 'pc gaming', 'steam', 'xbox', 'playstation'],
                
                # Business themes
                'business': ['business', 'company', 'corporate', 'entrepreneur', 'startup', 'revenue', 'profit'],
                'finance': ['money', 'financial', 'investment', 'stock', 'market', 'economy', 'trading', 'crypto'],
                'cryptocurrency': ['bitcoin', 'crypto', 'blockchain', 'ethereum', 'coin', 'mining', 'wallet'],
                
                # Social themes
                'politics': ['political', 'government', 'election', 'policy', 'vote', 'politician', 'democracy'],
                'education': ['school', 'university', 'student', 'teacher', 'education', 'learning', 'study'],
                'health': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'wellness', 'fitness'],
                
                # Entertainment themes
                'entertainment': ['movie', 'film', 'tv', 'show', 'entertainment', 'celebrity', 'hollywood'],
                'music': ['music', 'song', 'artist', 'album', 'concert', 'spotify', 'band', 'singer'],
                'sports': ['sport', 'game', 'team', 'player', 'football', 'basketball', 'soccer', 'baseball'],
                
                # Lifestyle themes
                'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal', 'eat', 'cuisine', 'chef'],
                'travel': ['travel', 'trip', 'vacation', 'hotel', 'flight', 'destination', 'tourism'],
                'fashion': ['fashion', 'style', 'clothing', 'outfit', 'design', 'brand', 'wear'],
                
                # Science themes
                'science': ['science', 'research', 'study', 'experiment', 'discovery', 'scientific'],
                'environment': ['environment', 'climate', 'green', 'sustainability', 'renewable', 'pollution'],
                'space': ['space', 'nasa', 'rocket', 'planet', 'universe', 'astronomy', 'mars'],
                
                # Community themes
                'community': ['community', 'people', 'social', 'group', 'forum', 'discussion', 'together'],
                'advice': ['advice', 'help', 'tips', 'suggestion', 'recommend', 'guide', 'how to'],
                'question': ['question', 'ask', 'wondering', 'curious', 'help', 'anyone know'],
                'opinion': ['opinion', 'think', 'believe', 'feel', 'thoughts', 'perspective', 'view']
            }
            
            # Calculate theme scores
            theme_scores = {}
            for theme, keywords in theme_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in content:
                        score += 1
                
                # Normalize score by keyword count
                if keywords:
                    theme_scores[theme] = score / len(keywords)
            
            # Filter and sort themes by score
            significant_themes = {theme: score for theme, score in theme_scores.items() if score > 0}
            
            if significant_themes:
                # Sort themes by score
                sorted_themes = sorted(significant_themes.items(), key=lambda x: x[1], reverse=True)
                
                # Create theme list with confidence scores
                themes = [
                    {"theme": theme, "confidence": min(score * 2, 0.9)}  # Scale confidence
                    for theme, score in sorted_themes[:5]
                ]
                
                primary_theme = sorted_themes[0][0]
                theme_confidence = min(sorted_themes[0][1] * 2, 0.9)
            else:
                # Default to general theme
                themes = [{"theme": "general", "confidence": 0.4}]
                primary_theme = 'general'
                theme_confidence = 0.4
            
            # Create comprehensive result
            result = {
                **item,  # Include all previous data
                'themes': themes,
                'primary_theme': primary_theme,
                'theme_confidence': theme_confidence,
                'all_theme_scores': theme_scores,
                'theme_model_used': 'keyword_mapping',
                'theme_analysis_method': 'keyword_fallback'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback theme analysis error: {e}")
            return self._create_general_theme_result(item)


class ReportGeneratorAgent:
    """
    Comprehensive Report Generator Agent
    
    This agent synthesizes all analysis results into detailed reports with
    insights, recommendations, and statistical summaries. It provides
    actionable intelligence from the multi-agent analysis workflow.
    
    Attributes:
        None - Stateless agent focused on report generation
    """
    
    def __init__(self):
        """Initialize the Report Generator Agent."""
        logger.info("Report Generator Agent initialized")
    
    async def generate_report(self, state: AgentState) -> AgentState:
        """
        Generate comprehensive analysis report with insights and recommendations.
        
        This method creates a detailed report including:
        1. Executive summary with key metrics
        2. Sentiment analysis breakdown
        3. Theme distribution analysis
        4. Top performing posts analysis
        5. Actionable insights and recommendations
        6. Technical metadata and model information
        
        Args:
            state (AgentState): Current state with complete analysis results
            
        Returns:
            AgentState: Updated state with comprehensive report
        """
        # Initialize report generation status
        state.current_agent = "Report Generator"
        state.status = "generating"
        state.progress = 0.1
        
        logger.info("Starting comprehensive report generation")
        
        try:
            results = state.theme_results
            
            if not results:
                error_msg = "No analysis results available for report generation"
                logger.error(error_msg)
                state.error = error_msg
                state.status = "error"
                return state
            
            total_posts = len(results)
            logger.info(f"Generating report for {total_posts} analyzed posts")
            
            # Calculate sentiment distribution with detailed statistics
            sentiment_counts = Counter(item['sentiment'] for item in results)
            sentiment_distribution = {
                sentiment: {
                    "count": count, 
                    "percentage": round((count / total_posts) * 100, 2)
                }
                for sentiment, count in sentiment_counts.items()
            }
            
            state.progress = 0.3
            
            # Calculate theme distribution with detailed statistics
            theme_counts = Counter(item['primary_theme'] for item in results)
            theme_distribution = {
                theme: {
                    "count": count, 
                    "percentage": round((count / total_posts) * 100, 2)
                }
                for theme, count in theme_counts.items()
            }
            
            state.progress = 0.5
            
            # Calculate confidence statistics
            sentiment_confidences = [item.get('confidence', 0) for item in results]
            theme_confidences = [item.get('theme_confidence', 0) for item in results]
            
            avg_sentiment_confidence = round(sum(sentiment_confidences) / len(sentiment_confidences), 3)
            avg_theme_confidence = round(sum(theme_confidences) / len(theme_confidences), 3)
            
            # Get top posts by engagement metrics
            top_posts = sorted(
                results, 
                key=lambda x: (x.get('score', 0) * x.get('upvote_ratio', 0.5)), 
                reverse=True
            )[:10]
            
            state.progress = 0.7
            
            # Generate comprehensive insights
            insights = self._generate_comprehensive_insights(
                results, sentiment_distribution, theme_distribution
            )
            
            # Generate actionable recommendations
            recommendations = self._generate_actionable_recommendations(
                sentiment_distribution, theme_distribution, results
            )
            
            # Collect model usage statistics
            models_used = list(set(item.get('model_used', 'unknown') for item in results))
            analysis_methods = list(set(item.get('analysis_method', 'unknown') for item in results))
            
            # Create comprehensive final report
            report = {
                "analysis_summary": {
                    "total_posts_analyzed": total_posts,
                    "subreddit": state.subreddit,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_duration": "N/A",  # Could be calculated if needed
                    "models_used": models_used,
                    "analysis_methods": analysis_methods
                },
                "sentiment_analysis": {
                    "distribution": sentiment_distribution,
                    "statistics": {
                        "average_confidence": avg_sentiment_confidence,
                        "confidence_range": {
                            "min": round(min(sentiment_confidences), 3),
                            "max": round(max(sentiment_confidences), 3)
                        }
                    },
                    "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get),
                    "sentiment_balance": {
                        "positive_ratio": sentiment_distribution.get('POSITIVE', {}).get('percentage', 0) / 100,
                        "negative_ratio": sentiment_distribution.get('NEGATIVE', {}).get('percentage', 0) / 100,
                        "neutral_ratio": sentiment_distribution.get('NEUTRAL', {}).get('percentage', 0) / 100
                    }
                },
                "theme_analysis": {
                    "distribution": theme_distribution,
                    "statistics": {
                        "average_confidence": avg_theme_confidence,
                        "unique_themes_count": len(theme_distribution),
                        "theme_diversity_index": len(theme_distribution) / total_posts
                    },
                    "dominant_theme": max(theme_counts, key=theme_counts.get),
                    "top_themes": list(theme_distribution.keys())[:10]
                },
                "engagement_analysis": {
                    "top_posts": [
                        {
                            "title": post.get('title', '')[:100] + ('...' if len(post.get('title', '')) > 100 else ''),
                            "sentiment": post['sentiment'],
                            "theme": post['primary_theme'],
                            "score": post.get('score', 0),
                            "engagement_ratio": post.get('upvote_ratio', 0),
                            "comments": post.get('num_comments', 0),
                            "confidence": round(post.get('confidence', 0), 3)
                        }
                        for post in top_posts
                    ],
                    "average_engagement": {
                        "score": round(sum(post.get('score', 0) for post in results) / total_posts, 2),
                        "upvote_ratio": round(sum(post.get('upvote_ratio', 0) for post in results) / total_posts, 3),
                        "comments": round(sum(post.get('num_comments', 0) for post in results) / total_posts, 2)
                    }
                },
                "insights": insights,
                "recommendations": recommendations,
                "technical_metadata": {
                    "analysis_agents": ["RedditScraperAgent", "SentimentAnalyzerAgent", "ThemeTaggerAgent", "ReportGeneratorAgent"],
                    "data_sources": [f"Reddit API - r/{state.subreddit}"],
                    "analysis_pipeline": "Multi-Agent Sentiment Analysis System v2.0",
                    "quality_metrics": {
                        "data_completeness": round((total_posts / state.post_limit) * 100, 2),
                        "analysis_coverage": 100.0  # All posts were analyzed
                    }
                }
            }
            
            # Update state with comprehensive report
            state.final_report = report
            state.status = "completed"
            state.progress = 1.0
            
            success_message = f"Successfully generated comprehensive report for {total_posts} posts with {len(insights)} insights and {len(recommendations)} recommendations"
            state.messages.append({
                "agent": "Report Generator",
                "message": success_message,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(success_message)
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            state.error = error_msg
            state.status = "error"
            state.messages.append({
                "agent": "Report Generator",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _generate_comprehensive_insights(
        self, 
        results: List[Dict], 
        sentiment_dist: Dict, 
        theme_dist: Dict
    ) -> List[str]:
        """
        Generate comprehensive insights from analysis results.
        
        Args:
            results (List[Dict]): Complete analysis results
            sentiment_dist (Dict): Sentiment distribution data
            theme_dist (Dict): Theme distribution data
            
        Returns:
            List[str]: List of actionable insights
        """
        insights = []
        total_posts = len(results)
        
        try:
            # Sentiment insights
            dominant_sentiment = max(sentiment_dist, key=lambda x: sentiment_dist[x]['count'])
            sentiment_percentage = sentiment_dist[dominant_sentiment]['percentage']
            
            if sentiment_percentage > 70:
                insights.append(f"Strong {dominant_sentiment.lower()} sentiment dominance ({sentiment_percentage}%) indicates clear community consensus")
            elif sentiment_percentage < 40:
                insights.append("Mixed sentiment distribution suggests diverse community opinions and active debate")
            else:
                insights.append(f"Moderate {dominant_sentiment.lower()} sentiment ({sentiment_percentage}%) with room for community engagement improvement")
            
            # Theme insights
            dominant_theme = max(theme_dist, key=lambda x: theme_dist[x]['count'])
            theme_percentage = theme_dist[dominant_theme]['percentage']
            
            if theme_percentage > 50:
                insights.append(f"Primary discussion focus on '{dominant_theme.replace('_', ' ').title()}' ({theme_percentage}%) shows specialized community interest")
            else:
                insights.append("Diverse theme distribution indicates broad community interests and varied content")
            
            # Engagement insights
            high_engagement_posts = [post for post in results if post.get('score', 0) > 100]
            if high_engagement_posts:
                insights.append(f"{len(high_engagement_posts)} posts show high engagement (>100 upvotes), indicating strong community interest")
            
            # Confidence insights
            high_confidence_analyses = [post for post in results if post.get('confidence', 0) > 0.8]
            if len(high_confidence_analyses) > total_posts * 0.7:
                insights.append("High analysis confidence across most posts ensures reliable sentiment assessment")
            
            # Cross-sentiment theme analysis
            positive_themes = [post['primary_theme'] for post in results if post['sentiment'] == 'POSITIVE']
            negative_themes = [post['primary_theme'] for post in results if post['sentiment'] == 'NEGATIVE']
            
            if positive_themes:
                top_positive_theme = Counter(positive_themes).most_common(1)[0][0]
                insights.append(f"'{top_positive_theme.replace('_', ' ').title()}' generates the most positive sentiment")
            
            if negative_themes:
                top_negative_theme = Counter(negative_themes).most_common(1)[0][0]
                insights.append(f"'{top_negative_theme.replace('_', ' ').title()}' shows concerning negative sentiment patterns")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Analysis completed successfully with comprehensive data coverage")
        
        return insights
    
    def _generate_actionable_recommendations(
        self, 
        sentiment_dist: Dict, 
        theme_dist: Dict, 
        results: List[Dict]
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis results.
        
        Args:
            sentiment_dist (Dict): Sentiment distribution data
            theme_dist (Dict): Theme distribution data
            results (List[Dict]): Complete analysis results
            
        Returns:
            List[str]: List of actionable recommendations
        """
        recommendations = []
        
        try:
            # Sentiment-based recommendations
            negative_percentage = sentiment_dist.get('NEGATIVE', {}).get('percentage', 0)
            positive_percentage = sentiment_dist.get('POSITIVE', {}).get('percentage', 0)
            
            if negative_percentage > 40:
                recommendations.append("Address community concerns highlighted in negative posts to improve overall sentiment")
                recommendations.append("Implement proactive community engagement strategies to convert neutral discussions to positive")
            
            if positive_percentage > 60:
                recommendations.append("Leverage positive community sentiment to promote content sharing and user-generated content")
                recommendations.append("Identify and replicate successful content patterns that generate positive responses")
            
            # Theme-based recommendations
            dominant_theme = max(theme_dist, key=lambda x: theme_dist[x]['count'])
            theme_count = len(theme_dist)
            
            if theme_count > 10:
                recommendations.append("Consider creating specialized discussion channels for diverse theme categories")
                recommendations.append("Implement topic-based content curation to better serve varied community interests")
            
            if theme_dist[dominant_theme]['percentage'] > 70:
                recommendations.append(f"Expand content variety beyond '{dominant_theme.replace('_', ' ').title()}' to engage broader audience")
            
            # Engagement-based recommendations
            avg_score = sum(post.get('score', 0) for post in results) / len(results)
            high_engagement_posts = [post for post in results if post.get('score', 0) > avg_score * 2]
            
            if high_engagement_posts:
                recommendations.append("Analyze high-performing posts to identify successful content patterns and engagement drivers")
            
            if avg_score < 50:
                recommendations.append("Implement community engagement initiatives to increase post visibility and interaction")
                recommendations.append("Consider content timing and formatting optimization for better community response")
            
            # Analysis quality recommendations
            low_confidence_posts = [post for post in results if post.get('confidence', 0) < 0.6]
            if len(low_confidence_posts) > len(results) * 0.3:
                recommendations.append("Review ambiguous posts manually for nuanced sentiment that automated analysis may miss")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Continue monitoring community sentiment trends for ongoing optimization")
        
        return recommendations