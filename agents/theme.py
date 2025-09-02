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
            raise Exception("HuggingFace transformers not available")
            
        try:
            # Load zero-shot classification model with truncation
            self.classification_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                truncation=True,
                max_length=512
            )
            
            self.model_loaded = True
            print("✅ Theme extraction model loaded: BART-large MNLI (with truncation)")
            
        except Exception as e:
            print(f"❌ Error loading theme models: {e}")
            raise e
    
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

class ThemeTaggerAgent:
    """Advanced Theme Analysis using Transformer Models and NLP"""
    
    def __init__(self):
        self.classification_pipeline = None
        self.tokenizer = None
        self.model_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load transformer models for theme classification"""
        if not HF_AVAILABLE:
            raise Exception("HuggingFace transformers not available")
            
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            
            # Load zero-shot classification model with proper tokenization
            self.classification_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                tokenizer=self.tokenizer,
                truncation=True,
                max_length=500  # Conservative limit
            )
            
            self.model_loaded = True
            print("✅ Theme extraction model loaded with tokenizer")
            
        except Exception as e:
            print(f"❌ Error loading theme models: {e}")
            raise e
    
    def _smart_truncate_tokens(self, text: str, max_tokens: int = 450) -> str:
        """Truncate text based on actual token count"""
        if not text or not text.strip():
            return ""
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text.strip()

    async def extract_themes(self, sentiment_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract themes using transformer-based classification"""
        
        if not self.model_loaded:
            raise Exception("Theme extraction models not loaded properly")
        
        # Enhanced candidate themes
        candidate_themes = [
            "product_quality", "customer_service", "pricing", "user_experience", 
            "performance", "features", "reliability", "innovation", "support",
            "bugs", "updates", "design", "usability", "value", "competition",
            "recommendation", "satisfaction", "problems", "improvement", "technology",
            "leadership", "business_strategy", "market_trends", "financial_performance",
            "social_impact", "controversy", "news", "announcement", "partnership"
        ]
        
        results = []
        
        for item in sentiment_results:
            raw_content = item.get('content', '').strip()
            
            if len(raw_content) < 10:
                print(f"Skipping very short content")
                themes = []
                theme_scores = {}
            else:
                try:
                    # Smart token-based truncation
                    content = self._smart_truncate_tokens(raw_content, 450)
                    
                    if not content:
                        print("Content became empty after token truncation")
                        themes = []
                        theme_scores = {}
                    else:
                        print(f"Processing for themes: {len(self.tokenizer.encode(content))} tokens")
                        
                        # Use zero-shot classification
                        classification_result = self.classification_pipeline(
                            content, 
                            candidate_themes,
                            multi_label=True
                        )
                        
                        # Get themes with confidence > 0.2
                        themes = [
                            label for label, score in zip(
                                classification_result['labels'], 
                                classification_result['scores']
                            ) if score > 0.2
                        ][:8]
                        
                        theme_scores = {
                            label: round(score, 4) for label, score in zip(
                                classification_result['labels'], 
                                classification_result['scores']
                            ) if score > 0.2
                        }
                        
                        print(f"✅ Extracted {len(themes)} themes via AI")
                    
                except Exception as e:
                    print(f"❌ Critical error in theme extraction: {e}")
                    raise e
            
            result = {
                'content': content if 'content' in locals() else raw_content,
                'original_content': raw_content,
                'user': item.get('user', ''),
                'date': item.get('date', datetime.datetime.now().isoformat()),
                'sentiment': item.get('sentiment', 'NEUTRAL'),
                'confidence': item.get('confidence', 0.5),
                'themes': themes,
                'theme_scores': theme_scores,
                'model_used': 'facebook/bart-large-mnli'
            }
            results.append(result)
            await asyncio.sleep(0.05)
        
        return results