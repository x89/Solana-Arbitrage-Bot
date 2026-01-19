import os
from dotenv import load_dotenv
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import date, timedelta

class TransformerSentimentAnalyzer:
    def __init__(
        self,
        model_name: str = 'tabularisai/multilingual-sentiment-analysis',
    ):
        """
        Initializes the transformer-based sentiment pipeline and sets up News API credentials.

        Args:
            model_name (str): Hugging Face model for sentiment analysis.
        """
        load_dotenv()
        self.api_key = os.getenv('NEWS_API_KEY')

        # Initialize Hugging Face sentiment-analysis pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.news_endpoint = 'https://newsapi.org/v2/everything'

    def fetch_articles(
        self,
        query: str = 'stock',
        page_size: int = 100,
        from_date: str = None,
        to_date: str = None
    ) -> list:
        """
        Fetches news articles matching the query for a given date range (defaults to today).

        Args:
            query (str): Search term for the News API (e.g., 'stock').
            page_size (int): Number of articles to retrieve (max 100).
            from_date (str): ISO date (YYYY-MM-DD) to start search (inclusive).
            to_date (str): ISO date (YYYY-MM-DD) to end search (inclusive).

        Returns:
            list: A list of article dicts.
        """
        # Default to today's date if not provided
        today_str = date.today().isoformat()
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()
        params = {
            'q': query,
            'pageSize': page_size,
            'apiKey': self.api_key,
            'language': 'en',
            'from': from_date or yesterday_str,
            'to': to_date or today_str,
        }
        response = requests.get(self.news_endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('articles', [])

    def analyze_sentiment(self, text: str) -> dict:
        """
        Runs transformer-based sentiment analysis on a single text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dict with 'label' and 'score'.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score_map = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
        scores = (probabilities * score_map).sum(dim=-1)
        return scores.tolist()

    def analyze_articles(self, articles: list) -> tuple:
        """
        Applies sentiment analysis to a list of articles.

        Args:
            articles (list): List of article dicts from fetch_articles().

        Returns:
            tuple: (List of results, average sentiment score or None)
        """
        results = []
        total_score = 0

        if not articles:
            return results, None

        for art in articles:
            title = art.get('title', '')
            desc = art.get('description', '') or ''
            text = f"{title}. {desc}"
            sentiment = self.analyze_sentiment(text)
            results.append({
                'title': title,
                'description': desc,
                'score': sentiment[0]
            })
            total_score += sentiment[0]

        avg_score = total_score / len(articles)
        return results, avg_score

if __name__ == '__main__':
    analyzer = TransformerSentimentAnalyzer()
    articles = analyzer.fetch_articles('Apple stock')  # typo fixed: 'APPL' → 'AAPL'
    report, sentiment = analyzer.analyze_articles(articles)

    if not report:
        print("No articles found.")
    else:
        for item in report:
            print(f"Title: {item['title']}")
            print(f"Description: {item['description']}")
            print(f"Sentiment: {item['score']:.2f}")
            print('-' * 80)
        print(f"Average sentiment: {sentiment:.2f}")
