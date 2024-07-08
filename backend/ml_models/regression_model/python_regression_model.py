import json
import os
# import joblib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# Initialize sklearn models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Custom Regression model
from linear_regression_model import linearRegressionModel
from random_forest_model import randomForestRegressionModel

filePath = os.path.join(os.path.dirname(__file__),'../json/evaluationMatrix.json')

def evaluationMatrix():
    # Specify the file name

    # Read JSON object from file
    with open(filePath, 'r') as file:
        data = json.load(file)

    return data


def trainModel():
    # Load CSV
    df = pd.read_csv('../dataset/post_process_dataset.csv')

    y = df['charges']
    x = df.drop(['charges'], axis=1)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Dictionary to store evaluation results
    evaluation_results = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # filePath = os.path.join(os.path.dirname(__file__), 'price_prediction_rfl.pkl')
        # joblib.dump(model, filePath)

        evaluation_results[name] = {
            "MAE": mae,
            "MSE": mse,
            "R²": r2
        }

        c_L_R_M = linearRegressionModel(df, 'charges', 0.001),
        C_R_F_R_M = randomForestRegressionModel(df, 'charges'),

        evaluation_results['customLinearRegressionModel'] = c_L_R_M[0]['customLinearRegressionModel']
        evaluation_results['customRandomForestRegressionModel'] = C_R_F_R_M[0]['customRandomForestRegressionModel']

    # Write JSON object to file
    with open(filePath, 'w') as file:
        json.dump(evaluation_results, file, indent=4)




trainModel()
print(evaluationMatrix())