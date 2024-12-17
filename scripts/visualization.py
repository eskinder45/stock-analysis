import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import os
import re
from scripts.qualitative import sentiment


# a function that returns a number of articles published by top n publishers
def articls_per_publisher(df:pd.DataFrame,top:int):
    articles_per_publisher_ = df['publisher'].value_counts()[:top]
    # Plot the data
    plt.figure(figsize=(10, 6))
    articles_per_publisher_.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Number of Articles Per Publisher', fontsize=16)
    plt.xlabel('Publisher', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


#a function to compute and plot word cloud of the headline column
def word_cloud(df:pd.DataFrame):
    # Combining all headlines into one corpus
    corpus = ' '.join(df['headline'].dropna().astype(str))
    # Generating  word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(corpus)
    # Ploting  the result
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Headlines', fontsize=16)
    plt.show()

# topics category published by to n publishers
def topic_per_publisher(df:pd.DataFrame,top:int):
    # Preprocess headlines
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    dtm = vectorizer.fit_transform(df['headline'].dropna().astype(str))
    # Load the LDA model from the file
    with open(os.path.dirname(os.getcwd()) + "\\models\\lda_model.pkl", 'rb') as file:
        lda = pickle.load(file)
    # finding the dominant topic for each article
    df['topic'] = lda.transform(dtm).argmax(axis=1)

    for count, i in enumerate(list(dict(df['publisher'].value_counts()[:top]).keys())):
        if count == 0:
            new_df  = df[df['publisher'] == i]
        else:
            tmp = df[df['publisher'] == i]
            df_lst = [new_df,tmp]
            new_df = pd.concat(df_lst)
    # Count of topics by publisher
    topics_by_publisher = new_df.groupby(['publisher', 'topic']).size().reset_index(name='count')
    # Plot the data
    plt.figure(figsize=(12, 8))
    sns.barplot(data=topics_by_publisher, x='topic', y='count', hue='publisher', dodge=True)
    plt.title('Topics Published by top 10 Publisher', fontsize=16)
    plt.xlabel('Topic', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.legend(title='Publisher', loc='upper right')
    plt.show()


#number of articles per topic
def articles_per_topic(df:pd.DataFrame):

    # Preprocess headlines
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    dtm = vectorizer.fit_transform(df['headline'].dropna().astype(str))
    # Load the LDA model from the file
    with open(os.path.dirname(os.getcwd()) + "\\models\\lda_model.pkl", 'rb') as file:
        lda = pickle.load(file)
    # finding the dominant topic for each article
    df['topic'] = lda.transform(dtm).argmax(axis=1)

    # Count the number of articles per topic
    topic_counts = df['topic'].value_counts().sort_index()

    # Plot the data
    plt.figure(figsize=(8, 6))
    topic_counts.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Frequency of Topics in Articles', fontsize=16)
    plt.xlabel('Topic', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def articles_by_email_domian(df:pd.DataFrame):
    df['domain'] = df['publisher'].apply(lambda x: re.search(r'@(\w+\.\w+)', x).group(1) if '@' in x else None)
    domain_counts = df['domain'].dropna().value_counts()
    plt.figure(figsize=(10, 6))
    domain_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Email Domain Analysis', fontsize=16)
    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



#number of articles to each sentiment by top n publishers
def sentiment_per_publisher(df:pd.DataFrame, top:int):
    result = sentiment(df)
    df['sentiment'] = result['sentiment']
    df['sentiment_category'] = result['sentiment_category']
    for count, i in enumerate(list(dict(df['publisher'].value_counts()[:top]).keys())):
        if count == 0:
            new_df  = df[df['publisher'] == i]
        else:
            tmp = df[df['publisher'] == i]
            df_lst = [new_df,tmp]
            new_df = pd.concat(df_lst)
    sentiments_by_publisher = new_df.groupby(['publisher', 'sentiment_category']).size().reset_index(name='count')
    plt.figure(figsize=(12, 8))
    sns.barplot(data=sentiments_by_publisher, x='sentiment_category', y='count', hue='publisher', dodge=True)
    plt.title('Sentiment Analysis Per Publisher', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.legend(title='Publisher', loc='upper right')
    plt.tight_layout()
    plt.show()


def article_per_day_of_week(df:pd.DataFrame):
    df['day_of_week'] = df['date'].dt.day_name()
    day_counts = df['day_of_week'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=day_counts.index, y=day_counts.values, palette='viridis')
    plt.title('Articles Published Per Day of the Week', fontsize=16)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



def article_per_hour(df:pd.DataFrame):
    df['hour'] = df['date'].dt.hour
    hour_counts = df['hour'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=hour_counts.index, y=hour_counts.values, marker='o', color='blue')
    plt.title('Articles Published Per Hour', fontsize=16)
    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid()
    plt.tight_layout()
    plt.show()


def articles_per_month(df:pd.DataFrame):
    df['month'] = df['date'].dt.month
    month_counts = df['month'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=month_counts.index, y=month_counts.values, palette='coolwarm')
    plt.title('Articles Published Per Month', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



#Publication Frequency Over Time
def publication_freq_wrt_t(df:pd.DataFrame):
    daily_articles = df.groupby(df['date'].dt.date)['headline'].count()
    daily_articles.plot(kind='line', title='Publication Frequency Over Time')
    plt.show()
