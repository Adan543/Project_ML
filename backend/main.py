from flask import Flask, request, jsonify, render_template
import joblib, os, json
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
filePath = os.path.join(os.path.dirname(__file__), 'price_prediction_rfl.pkl')
model = joblib.load(filePath)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    age = int(request.form['age'])
    gender = request.form['gender']
    bmi = float(request.form['bmi'])
    smoker = request.form['smoker']
    children = int(request.form['children'])
    region = request.form['region']

    # Preprocess the inputs
    gender = 1 if gender == 'male' else 0
    smoker = 1 if smoker == 'yes' else 0
    region_dict = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region = region_dict[region]

    # Create an array in the same order as the model was trained on
    input_data = np.array([[age, gender, bmi, smoker, children, region]])

    df = pd.DataFrame(input_data)
    price = model.predict(df)

    # Return the result
    # return render_template('index.html', prediction_text=f'Predicted Insurance Price: ${price[0]:.2f}')
    return jsonify({'price': price[0]})


@app.route('/getEvaluationMatrix', methods=['GET'])
def get_evaluation_matrix():
    with open('./ml_models/json/evaluationMatrix.json', 'r') as file:
        data = json.load(file)

    return data
    return jsonify(evaluation_matrix)

if __name__ == '__main__':
    app.run(debug=True)
