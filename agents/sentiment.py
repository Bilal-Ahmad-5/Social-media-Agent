from torch.jit import ignore
from transformers import pipeline
import numpy as np
from typing import List, Dict, Any, Optional
import datetime
from dataclasses import dataclass, field
from collections import Counter
from inspect import cleandoc
import warnings
import asyncio
import re
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

HF_AVAILABLE = True
print("HuggingFace Transformers loaded successfully!")

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
    
from torch.jit import ignore
from transformers import pipeline, AutoTokenizer
import numpy as np
from typing import List, Dict, Any, Optional
import datetime
from dataclasses import dataclass, field
from collections import Counter
from inspect import cleandoc
import warnings
import asyncio
import re
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

HF_AVAILABLE = True
print("HuggingFace Transformers loaded successfully!")

class SentimentAnalyzerAgent:
    """Advanced Sentiment Analysis using HuggingFace Transformer Models"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        self.sentiment_tokenizer = None
        self.emotion_tokenizer = None
        self.model_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained transformer models for sentiment analysis"""
        if not HF_AVAILABLE:
            raise Exception("HuggingFace transformers not available")
            
        try:
            # Load tokenizers first for proper token handling
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            
            # Load sentiment analysis model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer=self.sentiment_tokenizer,
                return_all_scores=True,
                truncation=True,
                max_length=510  # Leave room for special tokens
            )
            
            # Load emotion analysis model
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                tokenizer=self.emotion_tokenizer,
                return_all_scores=True,
                truncation=True,
                max_length=510  # Leave room for special tokens
            )
            
            self.model_loaded = True
            print("✅ Transformer models loaded with proper tokenization")
            
        except Exception as e:
            print(f"❌ Error loading transformer models: {e}")
            raise e
    
    def _smart_truncate(self, text: str, tokenizer, max_tokens: int = 500) -> str:
        """Intelligently truncate text based on actual token count"""
        if not text or not text.strip():
            return ""
        
        # Clean text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize to check actual token count
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= max_tokens:
            return text
        
        # If too long, decode back to get truncated text
        truncated_tokens = tokens[:max_tokens-1]  # Leave room for end token
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        # Try to end at a complete sentence
        sentences = re.split(r'[.!?]+', truncated_text)
        if len(sentences) > 1:
            # Remove the last incomplete sentence
            complete_text = '. '.join(sentences[:-1]) + '.'
            
            # Verify it's still within token limit
            final_tokens = tokenizer.encode(complete_text, add_special_tokens=True)
            if len(final_tokens) <= max_tokens:
                return complete_text
        
        return truncated_text

    async def analyze_sentiment(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment using transformer models"""
        
        if not self.model_loaded:
            raise Exception("Sentiment analysis models not loaded properly")
        
        results = []
        
        for post in posts:
            raw_content = post.get('title', '')
            raw_content += post.get('text', '')
            
            if not raw_content or not raw_content.strip():
                print("Skipping empty content")
                continue
            
            try:
                # Smart truncation based on actual tokens
                content = self._smart_truncate(raw_content, self.sentiment_tokenizer, 500)
                
                if not content:
                    print("Content became empty after truncation")
                    continue
                
                print(f"Processing content ({len(content)} chars, ~{len(self.sentiment_tokenizer.encode(content))} tokens)")
                
                # Get sentiment from RoBERTa model
                sentiment_result = self.sentiment_pipeline(content)[0]
                
                # Get emotion analysis (use same truncated content)
                emotion_content = self._smart_truncate(content, self.emotion_tokenizer, 500)
                emotion_result = self.emotion_pipeline(emotion_content)[0]
                
                # Process sentiment results
                sentiment_scores = {item['label']: round(item['score'], 4) for item in sentiment_result}
                emotion_scores = {item['label'].lower(): round(item['score'], 4) for item in emotion_result}
                
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
                    'original_content': raw_content,
                    'user': post.get('author', post.get('user', '')),
                    'date': post.get('created_utc', post.get('date', datetime.datetime.now().isoformat())),
                    'sentiment': mapped_sentiment,
                    'confidence': float(confidence),
                    'sentiment_scores': sentiment_scores,
                    'emotions': emotion_scores,
                    'model_used': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                    'tokens_used': len(self.sentiment_tokenizer.encode(content)),
                    'text_truncated': len(raw_content) != len(content)
                }
                results.append(result)
                
                print(f"✅ Sentiment: {mapped_sentiment} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"❌ Critical error in sentiment analysis: {e}")
                print(f"Raw content length: {len(raw_content)}")
                raise e
                
        return results