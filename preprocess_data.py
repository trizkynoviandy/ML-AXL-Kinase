import ast
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

def preprocess_data(file_path, threshold=0.95, test_size=0.2, random_state=1):
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    
    df = pd.read_csv(file_path)
    
    X = df.drop("class", axis=1)
    y = df["class"]
    
    class_dist = df.groupby("class").size()
    for index, val in class_dist.items():
        percentage = (val / sum(class_dist) * 100)
        print(f"Class {index} : {val} samples ({percentage:.2f}%)")
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y.values.ravel())
    
    variance_threshold = VarianceThreshold(0.1)
    variance_threshold.fit(X)
    X = X.iloc[:, variance_threshold.get_support()]

    def remove_collinear_features(x, threshold):
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        for i in iters:
            for j in range(i + 1):
                item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                if val >= threshold:
                    drop_cols.append(col.values[0])

        drops = set(drop_cols)
        x = x.drop(columns=drops)
        print('Removed Columns {}'.format(drops))
        return x

    X = remove_collinear_features(X, threshold)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test