from flask import Flask, render_template, request

import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Vectorize the input message
        message_vectorized = vectorizer.transform([message])

        # Make prediction using the loaded model
        prediction = model.predict(message_vectorized)

        result = "Spam" if prediction[0] == 1 else "Not Spam"

        return render_template('index.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
