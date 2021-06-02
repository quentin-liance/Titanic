import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask('titanic')

def predict_single(passenger, pipeline, model):
    X = pipeline.transform(passenger)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

with open('model_file.p', 'rb') as f_in:
    pipeline, model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    passenger_dict = request.get_json()
    passenger = pd.DataFrame.from_dict(data=passenger_dict, orient='index').T
    prediction = predict_single(passenger, pipeline, model)
    dead = prediction >= 0.5
   
    result = {
        'survival_probability': float(prediction),
        'survived': bool(dead)}
 
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)