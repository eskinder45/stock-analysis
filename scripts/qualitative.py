from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
from textblob import TextBlob


def headline_length(data):

    return data['headline'].apply(len)


def words_per_headline(data):
    return data['headline'].apply(lambda x: len(x.split()))



def stats(data):
    data['headline length'] = headline_length(data)
    data['word count per headline'] = words_per_headline(data)
    return data[['headline length','word count per headline']].describe()


def most_frequent_word(data,top):
    stop_words = set(stopwords.words('english'))
    all_words = ' '.join(data['headline']).lower().split()
    filtered_words = [word for word in all_words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top)

def articles_per_publisher(data):
    publisher_count = data.groupby('publisher')['headline'].count().reset_index()
    publisher_count = publisher_count.rename(columns={'headline': 'article_count'})
    return publisher_count.sort_values(by='article_count', ascending=False)



def sentiment(data):

    data['sentiment'] = data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['sentiment_category'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
    return data[['headline','sentiment','sentiment_category']]