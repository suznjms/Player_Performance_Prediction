import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import json


""" Function for Pearson Correlation """
def pearson_correlation(X,y,features_to_pick):
    print("\n\n\nPearson Correlation Detected\n\n\n")
    corr_list = []
    feature_list = list(X.columns)
    
    # finding the correlation values of each feature in feature list
    for f in feature_list:
        corr = np.corrcoef(X[f],y)[0,1]
        corr_list.append(corr)

    # if any NaN values present in corr_list, then make it as zero
    corr_list = [0 if np.isnan(c) else c for c in corr_list]

    # Ranking the features in corr_list and then picking the top 10 features
    corr_feature = X.iloc[:,np.argsort(np.abs(corr_list))[-features_to_pick:]].columns.tolist()
    corr_support = [True if f in corr_feature else False for f in feature_list]

    print("\n\n\nPearson Correlation Completed\n\n\n")

    return corr_feature, corr_support

""" Function for Chi - Squared """
def chi_squared(X, y, features_to_pick):
    print("\n\n\nChi - Squared Detected\n\n\n")
    X_normalize = MinMaxScaler().fit_transform(X)

    chi_square_func = SelectKBest(score_func=chi2, k=features_to_pick)
    chi_square_func.fit(X_normalize, y)

    chi_support = chi_square_func.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()

    print("\n\n\nChi - Squared Completed\n\n\n")

    return chi_support, chi_feature

""" Function for Recursive Feature Elimination """
def rfe(X, y, features_to_pick):
    print("\n\n\nRecursive Feature Elimination Detected\n\n\n")
    X_normalize = MinMaxScaler().fit_transform(X)

    rfe_func = RFE(estimator=LogisticRegression(), n_features_to_select=features_to_pick, step=10, verbose=5)
    rfe_func.fit(X_normalize,y)

    rfe_support = rfe_func.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()

    print("\n\n\nRecursive Feature Elimination Completed\n\n\n")

    return rfe_support, rfe_feature

""" Function for Embedded Logistic Regression - Lasso """
def embedded_log_reg(X, y, features_to_pick):
    print("\n\n\nEmbedded Logistic Regression - Lasso Detected\n\n\n")
    X_normalize = MinMaxScaler().fit_transform(X)

    log_reg_with_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)

    log_reg_func = SelectFromModel(estimator=log_reg_with_lasso, max_features=features_to_pick)
    log_reg_func.fit(X_normalize,y)

    log_reg_support = log_reg_func.get_support()
    log_reg_feature = X.loc[:,log_reg_support].columns.tolist()

    print("\n\n\nEmbedded Logistic Regression - Lasso Completed\n\n\n")

    return log_reg_support,log_reg_feature

""" Function for Embedded Random Forest """
def embedded_rf(X, y, features_to_pick):
    print("\n\n\nEmbedded Random Forest Detected\n\n\n")
    rf = RandomForestClassifier(n_estimators=200)

    rf_func = SelectFromModel(rf,max_features=features_to_pick)
    rf_func.fit(X,y)

    rf_support = rf_func.get_support()
    rf_feature = X.loc[:,rf_support].columns.tolist()

    print("\n\n\nEmbedded Random Forest Completed\n\n\n")

    return rf_support, rf_feature

""" Function for Embedded Light Gradient Boosting Method (Light GBM) """
def embedded_lgbm(X, y, features_to_pick):
    print("\n\n\nEmbedded Light Gradient Boosting Method Detected\n\n\n")
    lgbm = LGBMClassifier(n_estimators=500,
                           learning_rate=0.05, 
                           num_leaves=32, 
                           colsample_bytree=0.2, 
                           reg_alpha=3, 
                           reg_lambda=1, 
                           min_split_gain=0.01, 
                           min_child_weight=40)
    
    lgbm_func = SelectFromModel(lgbm,max_features=features_to_pick)
    lgbm_func.fit(X,y)

    lgbm_support = lgbm_func.get_support()
    lgbm_feature = X.loc[:,lgbm_support].columns.tolist()

    print("\n\n\nEmbedded Light Gradient Boosting Method Completed\n\n\n")

    return lgbm_support, lgbm_feature


""" Data Pre-processing """
def data_preprocessing(dataset_path):

    print("\n\n\ndata preprocessing started\n\n\n")

    football_data = pd.read_excel(dataset_path)

    num_columns = ['Age','Overall','Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',
       'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',
       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short passing', 'Shot power',
       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
       'Strength', 'Vision', 'Volleys']
    
    filtered_data = football_data[num_columns]
    filtered_data = filtered_data.dropna()
    filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce', downcast='integer')
    filtered_data = filtered_data.dropna()

    X=filtered_data.copy()
    del X['Overall']

    y=filtered_data['Overall']

    features_to_pick = 10

    print("\n\n\ndata preprocessing completed\n\n\n")

    return X, y, features_to_pick


""" Auto Feature Selector Function """
def feature_selector(dataset_path, methods=[]):

    print("\n\n\nAuto Feature Selector Function selected\n\n\n")

    X, y, features_to_pick = data_preprocessing(dataset_path)
    feature_name = list(X.columns)

    if 'pearson' in methods:
        print("\n\n\npearson detected\n\n\n")
        corr_feature, corr_support = pearson_correlation(X,y,features_to_pick)

    if 'chi-square' in methods:
        print("\n\n\nchi-squared detected\n\n\n")
        chi_support, chi_feature = chi_squared(X, y, features_to_pick)

    if 'rfe' in methods:
        print("\n\n\nrfe detected\n\n\n")
        rfe_support, rfe_feature = rfe(X, y, features_to_pick)

    if 'log-reg' in methods:
        print("\n\n\nlog-reg detected\n\n\n")
        log_reg_support, log_reg_feature = embedded_log_reg(X, y, features_to_pick)

    if 'rf' in methods:
        print("\n\n\nrf detected\n\n\n")
        rf_support, rf_feature = embedded_rf(X, y, features_to_pick)

    if 'lgbm' in methods:
        print("\n\n\nlgbm detected\n\n\n")
        lgbm_support, lgbm_feature = embedded_lgbm(X, y, features_to_pick)


    """ Combining & ranking all the features """
    pd.set_option('display.max_rows', None)

    features_df = pd.DataFrame({'Feature':feature_name, 'Pearson':corr_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':log_reg_support,
                                    'Random Forest':rf_support, 'LightGBM':lgbm_support})
    
    features_df['Total'] = np.sum(features_df, axis=1)

    features_df = features_df.sort_values(['Total','Feature'], ascending=False)
    features_df.index = range(1, len(features_df)+1)

    best_features = list(features_df['Feature'].head(10))

    with open ('output_file.txt','w') as f:
        for features in best_features:
            f.write(str(features) + '\n')

    print("\n\n\nAuto Feature Selector Function completed\n\n\n")
    return best_features, X, y
