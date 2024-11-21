from flask import Flask, render_template, request, redirect, url_for, send_file
import tweepy
import pandas as pd
from transformers import pipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twitter API credentials (Consider using environment variables for security)
API_KEY = os.getenv("TWITTER_API_KEY", "ZrzuWhnJYKhlJboy24wby5Lez")
API_SECRET = os.getenv("TWITTER_API_SECRET", "Tr7jCKop19cGsIF8HU5b0S5p77YLcaGKMqrbAWzYVyaG5IxIGf")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "1422117031277592578-CRfaoAJ81CrBVKtBt28rMAtYWsyWFG")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "GsYqxR5CMigM4KwDLc7Nw06zkueEuHhgSKQq2takP8YpC")

# Authenticate with Twitter API
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Initialize Hugging Face sentiment analysis pipeline with a specified model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Flask app initialization
app = Flask(__name__)

# Function to fetch healthcare-related tweets
def fetch_tweets(query, count=100):
    tweets_data = []
    try:
        for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(count):
            tweets_data.append(tweet.full_text)
    except Exception as e:
        print(f"Error fetching tweets: {e}")
    return tweets_data

# Function to analyze sentiment of tweets
def analyze_sentiments(tweets):
    results = []
    for tweet in tweets:
        try:
            sentiment = sentiment_analyzer(tweet)[0]
            results.append({"Tweet": tweet, "Sentiment": sentiment["label"], "Score": sentiment["score"]})
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
    return pd.DataFrame(results)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for processing the tweet query and displaying results
@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form['query']
    count = int(request.form['count'])
    
    # Fetching tweets based on user input
    tweets = fetch_tweets(query, count)
    
    if not tweets:
        return "No tweets found for your query. Please try again.", 400
    else:
        # Analyze sentiment of fetched tweets
        sentiment_results = analyze_sentiments(tweets)
        
        # Save the results to a CSV file
        output_file = "healthcare_tweets_sentiment.csv"
        sentiment_results.to_csv(output_file, index=False)
        
        # Return the file for download
        return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)