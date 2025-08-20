from flask import Flask,request,jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


with open('diabetes_model_flask.pkl','rb') as model_file:
  model=pickle.load(model_file)

with open('diabetes_scaler_flask.pkl','rb') as scaler_file:
  scaler=pickle.load(scaler_file)


@app.route('/')
def home():  # put application's code here
    return 'Diabetes Prediction App Is Running'

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.get_json()
        input_data=pd.DataFrame([data])

        if not data:
            return jsonify({'error': 'No data provided.'}),400

        required_data=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        if not all(col in input_data.columns for col in required_data):
            return jsonify({'error': 'Missing required columns.'}),400

        #predict result
        scaled_data=scaler.transform(input_data)
        prediction=model.predict(scaled_data)

        response={
            "prediction":"Diabetes" if prediction[0]==1 else "No Diabetes"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}),500






if __name__ == '__main__':
    app.run(debug=True)
