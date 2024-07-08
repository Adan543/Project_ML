import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os



class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_features=self.max_features, random_state=np.random.randint(0, 10000))
            X_sample, y_sample = resample(X, y, random_state=np.random.randint(0, 10000))
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


def randomForestRegressionModel(df, target):
    # Define features and target variable

    X = df.drop(target, axis=1).values
    y = df[target].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with preprocessing and the custom random forest regressor
    # pipeline = Pipeline(
    #     steps=[('regressor', CustomRandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42))])

    model = CustomRandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    filePath = os.path.join(os.path.dirname(__file__),'../json/price_prediction_rfl.pkl')
    joblib.dump(model, filePath)

    # Predict on the test set
    y_pred_test_rf = model.predict(X_test)

    # Calculate metrics for the test set
    mae_test_rf = mean_absolute_error(y_test, y_pred_test_rf)
    mse_test_rf = mean_squared_error(y_test, y_pred_test_rf)
    r2_test_rf = r2_score(y_test, y_pred_test_rf)
    evaluation_result = {}

    # Store results in a dictionary
    evaluation_result = {
        'customRandomForestRegressionModel': {
            "MAE": mae_test_rf,
            "MSE": mse_test_rf,
            "R²": r2_test_rf
        }

    }

    return evaluation_result


# df = pd.read_csv('../dataset/post_process_dataset.csv')
#
# evaluation_results = randomForestRegressionModel(df, 'charges')
