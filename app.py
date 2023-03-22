from flask import Flask, request, render_template, redirect, url_for, jsonify
import pickle
from model import SentimentRecommenderModel
import os
from flask import send_from_directory

app = Flask(__name__)

sentiment_rec_model = SentimentRecommenderModel()


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_recommendations():
    # get user_name from html
    user_name = request.form['userName']
    user_name = user_name.lower()
    # obtaining list of recommended products
    items = sentiment_rec_model.getSentimentBasedRecommendations(user_name)
    if not(items is None):
        print(f"retrieving items.Number of items...{len(items)}")
        print('*'*10)
        print(items)
        print('*' * 10)
        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()),
                               zip=zip)
    else:
        return render_template("index.html",
                               message=f"{user_name} Searched for recommendation is not available in our database, recommendations cann't be given "
                                       "try suggested user name!")


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)
