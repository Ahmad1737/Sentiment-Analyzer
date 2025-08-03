from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        review = request.form['reviewText']
        review_vector = vectorizer.transform([review])
        result = model.predict(review_vector)[0]
        prediction = 'Positive 😄' if result == 1 else 'Negative 😞'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


