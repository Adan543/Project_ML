import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('../dataset/post_process_dataset.csv')

# Define the hypothesis function
def hypothesis(X, beta):
    return np.dot(X, beta)

# Define the cost function
def compute_cost(X, y, beta):
    m = len(y)
    J = (1 / (2 * m)) * np.sum((hypothesis(X, beta) - y) ** 2)
    return J

# Define gradient descent
def gradient_descent(X, y, beta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        gradient = (1 / m) * np.dot(X.T, (hypothesis(X, beta) - y))
        beta = beta - learning_rate * gradient
        cost_history[i] = compute_cost(X, y, beta)

    return beta, cost_history




def linearRegressionModel(df, target, learning_rate):
    # Define features and target variable

    X = df.drop(target, axis=1).values
    y = df[target].values

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Add intercept term to X
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize parameters
    beta = np.zeros(X_train.shape[1])
    num_iterations = 10000

    # Train the model
    beta, cost_history = gradient_descent(X_train, y_train, beta, learning_rate, num_iterations)

    # Predictions
    y_pred_train = hypothesis(X_train, beta)
    y_pred_test = hypothesis(X_test, beta)

    # Calculate metrics for the test set
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # Store results in a dictionary
    evaluation_result = {}
    evaluation_result['customLinearRegressionModel'] = {
        "MAE": mae_test,
        "MSE": mse_test,
        "R²": r2_test
    }

    return evaluation_result


# df = pd.read_csv('../dataset/post_process_dataset.csv')
#
# evaluation_results = linearRegressionModel(df, 'charges',  0.001)
#
# print(evaluation_results)



