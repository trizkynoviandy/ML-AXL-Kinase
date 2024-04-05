from preprocess_data import preprocess_data

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, gamma, X_train, X_test, y_train, y_test):
    model = XGBClassifier(max_depth=int(max_depth),
                          learning_rate=learning_rate,
                          n_estimators=int(n_estimators),
                          min_child_weight=int(min_child_weight),
                          subsample=subsample,
                          colsample_bytree=colsample_bytree,
                          gamma=gamma,
                          random_state=1,
                          verbosity=0)
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_train, y_train, cv=10).mean()
    return score

pbounds = {'max_depth': (5, 25),
           'learning_rate': (0.01, 0.5),
           'n_estimators': (50, 150),
           'min_child_weight': (1, 10),
           'subsample': (0.5, 1),
           'colsample_bytree': (0.5, 1),
           'gamma': (0, 1)}

def optimize_xgb(file_path):
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    optimizer = BayesianOptimization(f=lambda max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, gamma: xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, gamma, X_train, X_test, y_train, y_test),
                                     pbounds=pbounds,
                                     random_state=0)
    optimizer.maximize(init_points=2, n_iter=5)
    
    # Get best parameters
    best_params = optimizer.max['params']
    
    # Train XGB classifier with best parameters
    clf = XGBClassifier(max_depth=int(best_params['max_depth']),
                        learning_rate=best_params['learning_rate'],
                        n_estimators=int(best_params['n_estimators']),
                        min_child_weight=int(best_params['min_child_weight']),
                        subsample=best_params['subsample'],
                        colsample_bytree=best_params['colsample_bytree'],
                        gamma=best_params['gamma'],
                        random_state=1,
                        verbosity=0)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred, digits=4))