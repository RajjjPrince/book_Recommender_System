# BookFinder 📚

**BookFinder** is a Flask-based web application that recommends books based on a user's selection. Using precomputed similarity scores and book data, it suggests books similar to the one you choose, making book discovery easy and fun.

---

## Features

- Display popular books with ratings and number of votes.
- Get book recommendations by entering a book name.
- Clean, user-friendly interface built with Flask and HTML templates.
- Simple setup with pre-trained models stored in `.pkl` files.

---

## Project Structure

BookFinder/
│
├── app.py # Main Flask application
├── model/ # Precomputed data and models
│ ├── popular.pkl
│ ├── pt.pkl
│ ├── books.pkl
│ └── similarity_score.pkl
├── templates/ # HTML templates
│ ├── index.html
│ └── recommend.html
└── static/ # Optional CSS/JS/images

