import numpy as np
from flask import Flask, request, jsonify
import pickle
from bot import BertClassifier, process
from flask_restful import Api

model = BertClassifier()
with open('Model/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Voice Assistant"

@app.route('/predict', methods=['POST'])

def predict():
    speech = request.json.get('speech')
    input_query = np.array([[speech]])
    result = process(input_query)
    return jsonify({'response': result})

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5000)
