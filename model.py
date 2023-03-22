import pickle
import pandas as pd
import numpy as np


class SentimentRecommenderModel:
    # defining path pickle variables
    ROOT_PATH = "pickle/"
    MODEL_NAME = "best-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
        self.data = pd.read_csv("Data/sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))

    # defining a method to get top20 recommendations for the user
    def getTop20RecommendationByUser(self, user_name):
        top_20_recomnd = list(self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
        return top_20_recomnd

    # defining a method to filter only top 5 recommendations based on sentiment analysis model
    def getSentimentBasedRecommendations(self, user_name):
        if user_name in self.user_final_rating.index:
            recommendations = list(
                self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(recommendations)]
            # transform the input data using tf-idf vectorizer that has been already built
            X = self.vectorizer.transform(
                filtered_data["reviews_text_cleaned"].values.astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(
                lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2)
            top5_products = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, top5_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
        else:
            print(f"We are not able to recommend for the User name {user_name}. its not available in our database, Please try for the suggested users")
            return None


