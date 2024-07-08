import pandas as pd
import os
import numpy as np
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample


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

def pridiction(sample_data):
    filePath = os.path.join(os.path.dirname(__file__),'../json/price_prediction_rfl.pkl')
    model = joblib.load(filePath)

    df = pd.DataFrame(sample_data)
    print('df => ', df)
    price = model.predict(df)
    print('------------->', price)
    return price




sample_data = {
    'age': [19],
    'gender': [0],
    'bmi': [27.9],
    'children': [0],
    'smoker': [1],
    'region': [1]
}


print(pridiction(sample_data))
