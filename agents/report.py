# from collections import Counter
# import asyncio
# import datetime

# HF_AVAILABLE = True

# class ReportGeneratorAgent:
#     """Advanced Insights Generator with comprehensive analytics"""
    
#     async def generate_report(self, state):
#         """Generate comprehensive analysis report"""
#         await asyncio.sleep(1)
        
#         subreddit = state.subreddit
#         posts = state.raw_posts
#         sentiment_results = state.sentiment_results
#         theme_results = state.theme_results
        
#         # Advanced sentiment analysis
#         sentiments = [r.get('sentiment', 'NEUTRAL') for r in sentiment_results]
#         sentiment_counts = Counter(sentiments)
#         total_posts = len(sentiments)
        
#         if total_posts == 0:
#             return self._empty_report(subreddit)
        
#         # Calculate percentages and scores
#         sentiment_percentages = {
#             sentiment: round((count / total_posts) * 100, 1)
#             for sentiment, count in sentiment_counts.items()
#         }
        
#         # Advanced sentiment scoring
#         positive_weight = sentiment_counts.get('POSITIVE', 0) * 1.0
#         neutral_weight = sentiment_counts.get('NEUTRAL', 0) * 0.5
#         negative_weight = sentiment_counts.get('NEGATIVE', 0) * -1.0
        
#         sentiment_score = round((positive_weight + neutral_weight + negative_weight) / total_posts, 2)
        
#         # Theme analysis
#         all_themes = []
#         for result in theme_results:
#             all_themes.extend(result.get('themes', []))
        
#         theme_counts = Counter(all_themes)
#         top_themes = dict(theme_counts.most_common(10))
        
#         # Reputation status
#         if sentiment_score > 0.5:
#             reputation = "Excellent"
#         elif sentiment_score > 0.2:
#             reputation = "Good"
#         elif sentiment_score > -0.2:
#             reputation = "Neutral"
#         elif sentiment_score > -0.5:
#             reputation = "Concerning"
#         else:
#             reputation = "Critical"
        
#         # Generate insights
#         insights = self._generate_advanced_insights(
#             sentiment_counts, sentiment_score, theme_counts, subreddit
#         )
        
#         # Generate recommendations
#         recommendations = self._generate_strategic_recommendations(
#             sentiment_counts, theme_counts, reputation, subreddit
#         )
        
#         # Compile comprehensive report
#         report = {
#             'subreddit': subreddit,
#             'analysis_timestamp': datetime.now().isoformat(),
#             'model_info': {
#                 'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest' if HF_AVAILABLE else 'advanced_keyword_analysis',
#                 'theme_model': 'facebook/bart-large-mnli' if HF_AVAILABLE else 'advanced_keyword_matching',
#                 'transformers_enabled': HF_AVAILABLE
#             },
#             'summary': {
#                 'total_tweets': total_posts,
#                 'sentiment_score': sentiment_score,
#                 'reputation_status': reputation,
#                 'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get)
#             },
#             'sentiment_analysis': {
#                 'distribution': dict(sentiment_counts),
#                 'percentages': sentiment_percentages,
#                 'confidence_scores': [r.get('confidence', 0) for r in sentiment_results]
#             },
#             'theme_analysis': {
#                 'top_themes': top_themes,
#                 'theme_coverage': len(top_themes),
#                 'most_discussed': max(theme_counts, key=theme_counts.get) if theme_counts else 'None'
#             },
#             'insights': insights,
#             'recommendations': recommendations,
#             'data_quality': {
#                 'sample_size': total_posts,
#                 'analysis_depth': 'comprehensive',
#                 'confidence_level': 'high' if HF_AVAILABLE else 'medium'
#             }
#         }
#         return report
    
#     def _generate_advanced_insights(self, sentiment_counts, sentiment_score, theme_counts, subreddit):
#         """Generate intelligent insights based on analysis"""
#         insights = []
#         total = sum(sentiment_counts.values())
        
#         if total == 0:
#             return ["Insufficient data for meaningful analysis"]
        
#         # Sentiment insights
#         positive_ratio = sentiment_counts.get('POSITIVE', 0) / total
#         negative_ratio = sentiment_counts.get('NEGATIVE', 0) / total
        
#         if positive_ratio > 0.6:
#             insights.append(f"{subreddit} enjoys strong positive sentiment with {positive_ratio:.1%} favorable mentions")
#         elif negative_ratio > 0.4:
#             insights.append(f"{subreddit} faces significant reputation challenges with {negative_ratio:.1%} negative sentiment")
        
#         # Theme insights
#         if theme_counts:
#             top_theme = max(theme_counts, key=theme_counts.get)
#             insights.append(f"'{top_theme.replace('_', ' ')}' is the primary discussion focus")
            
#             if 'customer_service' in theme_counts and theme_counts['customer_service'] > 2:
#                 insights.append("Customer service experiences are a significant factor in brand perception")
            
#             if 'bugs' in theme_counts or 'performance' in theme_counts:
#                 insights.append("Technical issues are impacting user satisfaction and require attention")
        
#         return insights
    
#     def _generate_strategic_recommendations(self, sentiment_counts, theme_counts, reputation, subreddit):
#         """Generate actionable strategic recommendations"""
#         recommendations = []
#         total = sum(sentiment_counts.values())
        
#         if total == 0:
#             return ["Increase brand visibility and social media presence"]
        
#         negative_ratio = sentiment_counts.get('NEGATIVE', 0) / total
        
#         # Reputation management
#         if reputation in ["Critical", "Concerning"]:
#             recommendations.append("Immediate crisis management required - address negative feedback promptly")
#             recommendations.append("Implement proactive customer outreach and issue resolution protocols")
#         elif reputation == "Good":
#             recommendations.append("Leverage positive sentiment in marketing campaigns and testimonials")
        
#         # Theme-based recommendations
#         if 'customer_service' in theme_counts:
#             recommendations.append("Invest in customer service training and response time improvements")
        
#         if 'pricing' in theme_counts:
#             recommendations.append("Conduct competitive pricing analysis and communicate value proposition clearly")
        
#         if 'bugs' in theme_counts or 'performance' in theme_counts:
#             recommendations.append("Prioritize product quality improvements and technical stability")
        
#         if 'features' in theme_counts:
#             recommendations.append("Enhance user education about existing features and gather feature requests")
        
#         # General recommendations
#         recommendations.append("Establish continuous social media monitoring for trend identification")
#         recommendations.append("Develop content strategy targeting identified discussion themes")
        
#         return recommendations
    
#     def _empty_report(self, subreddit):
#         """Generate report when no data is available"""
#         return {
#             'subreddit': subreddit,
#             'analysis_timestamp': datetime.now().isoformat(),
#             'summary': {'total_tweets': 0, 'sentiment_score': 0, 'reputation_status': 'Unknown'},
#             'insights': ["No social media mentions found for analysis"],
#             'recommendations': ["Increase brand visibility and social media engagement"]
#         }

from collections import Counter
import asyncio
import datetime

HF_AVAILABLE = True

class ReportGeneratorAgent:
    """Advanced Insights Generator with comprehensive analytics"""
    
    async def generate_report(self, state):
        """Generate comprehensive analysis report"""
        await asyncio.sleep(1)
        
        # Handle both dict and object state formats
        if isinstance(state, dict):
            subreddit = state.get('subreddit', '')
            posts = state.get('raw_posts', [])
            sentiment_results = state.get('sentiment_results', [])
            theme_results = state.get('theme_results', [])
        else:
            subreddit = getattr(state, 'subreddit', '')
            posts = getattr(state, 'raw_posts', [])
            sentiment_results = getattr(state, 'sentiment_results', [])
            theme_results = getattr(state, 'theme_results', [])
        
        # Advanced sentiment analysis
        sentiments = [r.get('sentiment', 'NEUTRAL') for r in sentiment_results]
        sentiment_counts = Counter(sentiments)
        total_posts = len(sentiments)
        
        if total_posts == 0:
            return self._empty_report(subreddit)
        
        # Calculate percentages and scores
        sentiment_percentages = {
            sentiment: round((count / total_posts) * 100, 1)
            for sentiment, count in sentiment_counts.items()
        }
        
        # Advanced sentiment scoring
        positive_weight = sentiment_counts.get('POSITIVE', 0) * 1.0
        neutral_weight = sentiment_counts.get('NEUTRAL', 0) * 0.5
        negative_weight = sentiment_counts.get('NEGATIVE', 0) * -1.0
        
        sentiment_score = round((positive_weight + neutral_weight + negative_weight) / total_posts, 2)
        
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
            sentiment_counts, sentiment_score, theme_counts, subreddit
        )
        
        # Generate recommendations
        recommendations = self._generate_strategic_recommendations(
            sentiment_counts, theme_counts, reputation, subreddit
        )
        
        # Compile comprehensive report
        report = {
            'subreddit': subreddit,
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'model_info': {
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest' if HF_AVAILABLE else 'advanced_keyword_analysis',
                'theme_model': 'facebook/bart-large-mnli' if HF_AVAILABLE else 'advanced_keyword_matching',
                'transformers_enabled': HF_AVAILABLE
            },
            'summary': {
                'total_tweets': total_posts,
                'sentiment_score': sentiment_score,
                'reputation_status': reputation,
                'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else 'NEUTRAL'
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
                'sample_size': total_posts,
                'analysis_depth': 'comprehensive',
                'confidence_level': 'high' if HF_AVAILABLE else 'medium'
            }
        }
        return report
    
    def _generate_advanced_insights(self, sentiment_counts, sentiment_score, theme_counts, subreddit):
        """Generate intelligent insights based on analysis"""
        insights = []
        total = sum(sentiment_counts.values())
        
        if total == 0:
            return ["Insufficient data for meaningful analysis"]
        
        # Sentiment insights
        positive_ratio = sentiment_counts.get('POSITIVE', 0) / total
        negative_ratio = sentiment_counts.get('NEGATIVE', 0) / total
        
        if positive_ratio > 0.6:
            insights.append(f"{subreddit} enjoys strong positive sentiment with {positive_ratio:.1%} favorable mentions")
        elif negative_ratio > 0.4:
            insights.append(f"{subreddit} faces significant reputation challenges with {negative_ratio:.1%} negative sentiment")
        
        # Theme insights
        if theme_counts:
            top_theme = max(theme_counts, key=theme_counts.get)
            insights.append(f"'{top_theme.replace('_', ' ')}' is the primary discussion focus")
            
            if 'customer_service' in theme_counts and theme_counts['customer_service'] > 2:
                insights.append("Customer service experiences are a significant factor in brand perception")
            
            if 'bugs' in theme_counts or 'performance' in theme_counts:
                insights.append("Technical issues are impacting user satisfaction and require attention")
        
        return insights
    
    def _generate_strategic_recommendations(self, sentiment_counts, theme_counts, reputation, subreddit):
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
    
    def _empty_report(self, subreddit):
        """Generate report when no data is available"""
        return {
            'subreddit': subreddit,
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'summary': {'total_tweets': 0, 'sentiment_score': 0, 'reputation_status': 'Unknown'},
            'insights': ["No social media mentions found for analysis"],
            'recommendations': ["Increase brand visibility and social media engagement"]
        }