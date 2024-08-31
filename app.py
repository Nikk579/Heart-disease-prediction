from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and the scaler
model = pickle.load(open('ada_boost_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the label encoding mappings
label_mappings = {
    'Sex': {'F': 0, 'M': 1},
    'ChestPainType': {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3},
    'RestingECG': {'LVH': 0, 'Normal': 1, 'ST': 2},
    'ExerciseAngina': {'N': 0, 'Y': 1},
    'ST_Slope': {'Down': 0, 'Flat': 1, 'Up': 2}
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and process form data
        input_data = {
            'Age': request.form.get('age'),
            'Sex': request.form.get('sex'),
            'ChestPainType': request.form.get('chest_pain_type'),
            'RestingBP': request.form.get('resting_bp'),
            'Cholesterol': request.form.get('cholesterol'),
            'FastingBS': request.form.get('fasting_bs'),
            'RestingECG': request.form.get('resting_ecg'),
            'MaxHR': request.form.get('max_hr'),
            'ExerciseAngina': request.form.get('exercise_angina'),
            'Oldpeak': request.form.get('oldpeak'),
            'ST_Slope': request.form.get('st_slope')
        }

        # Check for empty fields
        errors = []
        for key, value in input_data.items():
            if not value:
                errors.append(f"{key.replace('_', ' ').title()} is required.")
        
        if errors:
            return render_template('index.html', 
                                   prediction_text='',
                                   input_data=input_data,
                                   errors=errors)

        # Convert to appropriate types
        try:
            input_data['Age'] = float(input_data['Age'])
            input_data['RestingBP'] = float(input_data['RestingBP'])
            input_data['Cholesterol'] = float(input_data['Cholesterol'])
            input_data['FastingBS'] = float(input_data['FastingBS'])
            input_data['MaxHR'] = float(input_data['MaxHR'])
            input_data['Oldpeak'] = float(input_data['Oldpeak'])
        except ValueError:
            return render_template('index.html', 
                                   prediction_text='',
                                   input_data=input_data,
                                   errors=['Numeric values are required for certain fields.'])

        # Convert categorical inputs using the provided mappings
        try:
            for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
                if input_data[col] not in label_mappings[col]:
                    raise ValueError(f"Unexpected value '{input_data[col]}' for {col}")
                input_data[col] = label_mappings[col][input_data[col]]
        except ValueError as ve:
            return render_template('index.html', 
                                   prediction_text='',
                                   input_data=input_data,
                                   errors=[str(ve)])

        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply StandardScaler to the input data
        input_df_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_df_scaled)

        # Convert prediction to a readable format
        output = 'Positive' if prediction[0] == 1 else 'Negative'

        return render_template('index.html', 
                               prediction_text=f'Heart Disease Prediction: {output}',
                               input_data=input_data)

    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error: {str(e)}',
                               input_data=input_data)


if __name__ == "__main__":
    app.run(debug=True)
