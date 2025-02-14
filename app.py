from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.joblib')

# Define feature names
feature_names = [
    'infant', 'water', 'budget', 'physician', 'salvador',
    'religious', 'satellite', 'aid', 'missile', 'immigration',
    'synfuels', 'education', 'superfund', 'crime', 'dutyfree_exports'
]

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the request
        features = [int(request.form.get(feature, 0)) for feature in feature_names]
        
        # Convert features to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)[0]
        
        # Prepare the response
        response = {
            'prediction': 'Republican' if prediction[0] == 1 else 'Democrat',
            'probability': {
                'democrat': float(probability[0]),
                'republican': float(probability[1])
            }
        }
        
        return render_template('result.html', result=response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

