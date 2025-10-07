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


---

## Setup Instructions

### 1. Clone the repository
  ```bash
  git clone https://github.com/your-username/BookFinder.git
  cd BookFinder

###2. Create a virtual environment

python -m venv venv


3. Activate the virtual environment
Windows:

venv\Scripts\activate
Mac/Linux:


source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

Flask
numpy
pandas
pickle-mixin
5. Run the Flask app

python app.py
6. Open in browser
Navigate to http://127.0.0.1:5000 to use BookFinder.

How to Use
Open the homepage to browse popular books.

Go to the Recommend page.

Enter the name of a book in the search field.

View recommended books with title, author, and image.

Contributing
Fork the repository.

Create a new branch:

git checkout -b feature-branch
Make changes and commit:


git commit -m "Add feature"
Push to your branch:

git push origin feature-branch
Open a Pull Request.
