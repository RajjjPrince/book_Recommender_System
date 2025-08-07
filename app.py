from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pickle files
with open("model/pt.pkl", "rb") as f:
    pt = pickle.load(f)

with open("model/similarity_score.pkl", "rb") as f:
    similarity_score = pickle.load(f)

# Recommendation function
def recommend(book_name):
    if book_name not in pt.index:
        return None
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    recommended_books = [pt.index[i[0]] for i in similar_items]
    return recommended_books

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_route():
    book_name = request.form.get('book')
    recommendations = recommend(book_name)
    if recommendations:
        return render_template('index.html', recommendations=recommendations)
    else:
        return render_template('index.html', error="Book not found in the dataset.")

if __name__ == '__main__':
    app.run(debug=True)
