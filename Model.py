""" Importing function from Best_Feature_Selector.py file """
from Best_Feature_Selector import feature_selector, data_preprocessing

""" Importing Dependencies """
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
import joblib
#from sklearn.externals import joblib



""" Training our ML model & hyper-tuning it """
def model(dataset_path, methods):
    print("\n\n\nmodel started\n\n\n")
    best_features, X, y = feature_selector(dataset_path, methods)
    print(best_features)
    X = X[best_features]
    

    # splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    

    # normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # parameter grid for hyper-tuning 
    param_grid = {'alpha': [0.01, 0.1, 1, 10]}

    # use of GridSearchCV
    gridsearch = GridSearchCV(estimator=Ridge(), param_grid=param_grid, cv=5, scoring='accuracy')
    gridsearch.fit(X_train, y_train)

    # getting the value of alpha
    best_alpha = gridsearch.best_params_['alpha']

    tuned_model = Ridge(alpha=best_alpha)
    tuned_model.fit(X_train,X_train)

    y_pred = tuned_model.predict(X_test)
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #print(f"Accuracy : {rmse}")
    print("\n\n\nmodel done\n\n\n")

    return tuned_model

""" Function to save any trained ML model """
def save_model(model, filename):
    joblib.dump(model,filename)

""" Function to load a saved ML model"""
def load_model(filename):
    loaded_model = joblib.load(filename)
    return loaded_model

def run_model(dataset_path, methods):
    trained_model = model(dataset_path,methods)
    model_filename = 'ridge_model.pkl'
    save_model(trained_model,model_filename)

print("Program has started")
dataset_path = 'CompleteDataset.xlsx'
methods = ['pearson','chi-square','rfe','log-reg','rf','lgbm']
run_model(dataset_path,methods)
