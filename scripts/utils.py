import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def data_loader(file_path):
    df = pd.read_csv(file_path)
    return df 

def headline_text_processing(df:pd.DataFrame):
    # Preprocess headlines
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    dtm = vectorizer.fit_transform(df['headline'].dropna().astype(str))
    return dtm
