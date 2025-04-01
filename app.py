from flask import Flask, request, jsonify, render_template

import joblib
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

# Load your trained stacked model
with open('stacked_model_tunedd.pkl', 'rb') as model_file:
    stacked_model = joblib.load('stacked_model_tunedd.pkl')

# Define the feature names (same order as the model expects)
feature_names = [
    'Current_assets', 'Cost_of_goods_sold', 'Depreciation_and_amortization', 
    'EBITDA', 'Inventory', 'Net_Income', 'Total_Receivables', 
    'Market_Value', 'Net_Sales', 'Total_Assets', 'Total_Long_term_Debt', 
    'EBIT', 'Gross_Profit', 'Total_Current_Liabilities', 'Retained_Earnings', 
    'Total_Revenue', 'Total_Liabilities', 'Total_Operating_Expenses'
]

@app.route('/')

import numpy as np
import shap

app = Flask(__name__)

# Load the saved model
loaded_model = joblib.load('stacked_model_tunedd.pkl')

# Specify the feature names
feature_names = ['Current assets', 'Cost of goods sold', 'Depreciation and amortization',
       'EBITDA', 'Inventory', 'Net Income', 'Total Receivables',
       'Market Value', 'Net Sales', 'Total Assets', 'Total Long-term Debt',
       'EBIT', 'Gross Profit', 'Total Current Liabilities',
       'Retained Earnings', 'Total Revenue', 'Total Liabilities',
       'Total Operating Expenses']

# Load the shap explainer
explainer = shap.TreeExplainer(loaded_model, feature_names=feature_names)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json
        
        # Ensure all required input fields are present
        required_fields = feature_names
        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Convert input data to numeric format
        for field, value in input_data.items():
            try:
                input_data[field] = float(value) 
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz

                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                v
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                abcd efg hi jk lmno op q rst u v w xyz
                v
                abcd efg hi jk lmno op q rst u v w xyz
            except ValueError:
                return jsonify({'error': f'Invalid value for field {field}: {value}'}), 400
        
        # Call the loaded model to make prediction
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        prediction_result = loaded_model.predict(input_array)
        
        # Map prediction result to class labels indicating bankruptcy or non-bankruptcy
        prediction_label = "Bankrupt" if prediction_result[0] == 1 else "Not Bankrupt"
        
        # If prediction is bankrupt, get feature importance for bankrupt prediction
        if prediction_result[0] == 1:
            shap_values = explainer.shap_values(input_array)
            feature_importance = {feature_names[i]: shap_values[0][i] for i in range(len(feature_names))}
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return jsonify({'result': prediction_label, 'feature_importance': sorted_importance}), 200
        else:
            # If prediction is not bankrupt, provide general advice
            return jsonify({'result': prediction_label, 'advice': 'Ensure stable earnings, low debt ratio, and healthy net income to total assets ratio.'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the POST request
    data = request.get_json()

    # Extract and convert inputs into a NumPy array for prediction
    input_features = [float(data[f'_{feature}']) for feature in feature_names]
    input_array = np.array(input_features).reshape(1, -1)

    # Make prediction using the loaded model
    prediction = stacked_model.predict(input_array)[0]
    prediction_proba = stacked_model.predict_proba(input_array)[0]

    # Initialize LIME explainer with current user input as the reference
    explainer = LimeTabularExplainer(
        training_data=np.array([input_array[0]]),  # The user's input is used here as the reference
        feature_names=feature_names,
        class_names=['Not Default', 'Default'],
        discretize_continuous=True
    )

    # Use LIME to explain the prediction
    explanation = explainer.explain_instance(
        input_array[0],  # The input data to explain
        stacked_model.predict_proba,  # The function to explain (model's probability prediction)
        num_features=5  # Adjust the number of features you want to explain
    )
    explanation_list = explanation.as_list()

    # Optional: provide advice based on prediction (customizable)
    advice = "Action needed!" if prediction == 1 else "All looks good."

    # Return prediction result and explanation to the frontend
    return jsonify({
        'result': int(prediction),
        'probability': prediction_proba.tolist(),
        'advice': advice,
        'feature_importance': explanation_list
    })

if __name__ == '__main__':
    app.run(debug=True)
