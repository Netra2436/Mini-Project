# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('best_model.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        if request.content_type == 'application/json':
            data = request.json
            features = list(data.values())
        else:  # Form data
            features = []
            for key in request.form:
                if key != 'submit':  # Skip the submit button
                    features.append(float(request.form[key]))
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Convert prediction to class label
        prediction_label = le.inverse_transform(prediction)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(features)[0]
        
        # Pair class names with probabilities
        classes = le.classes_
        class_probabilities = {le.inverse_transform([i])[0]: round(float(prob) * 100, 2) 
                               for i, prob in enumerate(probabilities)}
        
        # Sort by probability (highest first)
        sorted_probs = sorted(class_probabilities.items(), 
                     key=lambda x: x[1], 
                     reverse=True)
        print("Sorted Probabilities:", sorted_probs)

        return render_template('result.html', 
                              prediction=prediction_label,
                              probabilities=sorted_probs)
    
    except Exception as e:
        return jsonify({'error': str(e)})


# API endpoint for programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        features = list(data.values())
        features = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features)
        prediction_label = le.inverse_transform(prediction)[0]
        
        probabilities = model.predict_proba(features)[0]
        classes = le.classes_
        class_probabilities = {le.inverse_transform([i])[0]: float(prob) 
                               for i, prob in enumerate(probabilities)}
        
        return jsonify({
            'prediction': prediction_label,
            'probabilities': class_probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Make sure the templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True)