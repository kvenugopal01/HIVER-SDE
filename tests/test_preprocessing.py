import pytest
import pandas as pd
from src.data_preprocessing import clean_text, parse_twitter_date, preprocess_tweets

def test_clean_text():
    raw = "Hello &amp; welcome to &lt;AppleSupport&gt;!   "
    cleaned = clean_text(raw)
    assert cleaned == "Hello & welcome to <AppleSupport>!"

def test_parse_twitter_date():
    date_str = "Tue Oct 31 22:10:47 +0000 2017"
    dt = parse_twitter_date(date_str)
    assert dt.year == 2017
    assert dt.month == 10
    assert dt.day == 31

def test_preprocess_tweets():
    sample_data = pd.DataFrame([
        {
            "tweet_id": 1,
            "author_id": "user1",
            "inbound": True,
            "created_at": "Tue Oct 31 22:10:47 +0000 2017",
            "text": "My phone broke &amp; won't start"
        },
        {
            "tweet_id": 1, # duplicate ID
            "author_id": "user1",
            "inbound": True,
            "created_at": "Tue Oct 31 22:10:47 +0000 2017",
            "text": "My phone broke &amp; won't start"
        }
    ])
    processed = preprocess_tweets(sample_data)
    assert len(processed) == 1
    assert processed.iloc[0]['text_clean'] == "My phone broke & won't start"
