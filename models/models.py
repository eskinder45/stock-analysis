from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scripts.utils import headline_text_processing
import pandas as pd
import pickle
import os



def train_model(df:pd.DataFrame,components:int):
    lda = LatentDirichletAllocation(n_components=components, random_state=42) 
    lda.fit(headline_text_processing(df))
    # Save the LDA model to a file
    with open(os.path.dirname(os.getcwd()) + "\\models\\lda_model.pkl", 'wb') as file:
        pickle.dump(lda, file)



def predict(df:pd.DataFrame):
    # Load the LDA model from the file
    with open(os.path.dirname(os.getcwd()) + "\\models\\lda_model.pkl", 'rb') as file:
        lda = pickle.load(file)
    # Get the dominant topic for each article
    df['topic'] = lda.transform(headline_text_processing(df)).argmax(axis=1)
    return df