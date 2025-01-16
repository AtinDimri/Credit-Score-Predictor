from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['INCOME'], data['DEBT'], data['SAVINGS']]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'CREDIT_SCORE': prediction})

if __name__ == '__main__':
    app.run(debug = True)