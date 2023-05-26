# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:09:40 2023

@author: charl
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_test = pd.read_csv('C:/Users/charl/OneDrive/Documents/VU vakken/DMT/vu-dmt-assigment-2-2023/test1.csv', sep=',', header=0)
df_train_book = pd.read_csv('C:/Users/charl/OneDrive/Documents/VU vakken/DMT/vu-dmt-assigment-2-2023/clean2.csv', sep=',', header=0)
df_train = pd.read_csv('C:/Users/charl/OneDrive/Documents/VU vakken/DMT/vu-dmt-assigment-2-2023/clean4.csv', sep=',', header=0)
df_train['booking_bool']=df_train_book['booking_bool']

df_train
# for the columns prop_starrating and prop_review_score, a 0 represents a missing value and should be replaced by nan
df_train['prop_starrating'] = df_train['prop_starrating'].replace(0, np.nan)
df_train['prop_review_score'] = df_train['prop_review_score'].replace(0, np.nan)
df_train["prop_log_historical_price"] = df_train["prop_log_historical_price"].replace(0, np.nan)
df_train["orig_destination_distance"] = df_train["orig_destination_distance"].replace(0, np.nan)

# for the columns prop_starrating and prop_review_score, a 0 represents a missing value and should be replaced by nan
df_test['prop_starrating'] = df_test['prop_starrating'].replace(0, np.nan)
df_test['prop_review_score'] = df_test['prop_review_score'].replace(0, np.nan)
df_test["prop_log_historical_price"] = df_test["prop_log_historical_price"].replace(0, np.nan)
df_test["orig_destination_distance"] = df_test["orig_destination_distance"].replace(0, np.nan)



# from fancyimpute import IterativeImputer

# columns_with_missing = ['prop_review_score']

# # Create a copy of the data to preserve the original
# df_imputed = df_train.copy()

# # Perform MICE imputation
# imputer = IterativeImputer()
# df_imputed[columns_with_missing] = imputer.fit_transform(df_imputed[columns_with_missing])

# # Count the number of missing values in each column
# missing_values = df_train.isna().sum()

# df_train['prop_review_score'] = df_imputed['prop_review_score']
# # Count the number of missing values in each colum

# columns_with_missing = ['prop_starrating']

# # Create a copy of the data to preserve the original
# df_imputed = df_train.copy()

# # Perform MICE imputation
# imputer = IterativeImputer()
# df_imputed[columns_with_missing] = imputer.fit_transform(df_imputed[columns_with_missing])

# # Count the number of missing values in each column
# missing_values = df_train.isna().sum()

# df_train['prop_starrating'] = df_imputed['prop_starrating']
# # Count the number of missing values in each colum




missing_values = df_train.isna().sum()

# Calculate the percentage of missing values in each column
percent_missing = (missing_values / len(df_train)) * 100
 


#df_train = df_train.drop(['min_rate', 'max_rate'], axis=1)

df_test1=df_test.iloc[:1000]
df_train2=df_train.iloc[:1000]

df_train_unique = df_train[['prop_country_id', 'prop_id','booked_rank']].drop_duplicates()

df_test1 = df_test1.merge(df_train_unique, on=['prop_country_id', 'prop_id'], how='left')
 
#df_test2 = df_test1.merge(df_train[['prop_country_id','prop_id','booked_rank','booked_percent','click_percent','total_booked','total_clicked','search_count']], 
 #                               on=['prop_country_id', 'prop_id'], how='left')

##-------------------------------------------------------------------------------------------------------------------------------------------------
# # Step 1: Prepare the dataset
# X = df_train2.drop(['booking_bool', 'random_bool', 'click_bool','position'], axis=1)
# y = df_train2['booking_bool']


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df_test1=df_test.iloc[:10000]
df_train2=df_train.iloc[:10000]

# df_test1=df_test
# df_train2=df_train


df_train_unique = df_train[['prop_country_id', 'prop_id', 'booked_rank']].drop_duplicates()

df_test1 = df_test1.merge(df_train_unique, on=['prop_country_id', 'prop_id'], how='left')

X_train, X_test, y_train, y_test = train_test_split(
    df_train2.drop(['booking_bool','position','click_bool', 'max_rate', 'min_rate'], axis=1),
    df_train2['booking_bool'],
    test_size=0.2, random_state=42, stratify=df_train2['booking_bool']
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# def split_data_with_queries(df, test_size, random_state):
#     unique_queries = df['srch_id'].unique()
#     train_queries, test_queries = train_test_split(unique_queries, test_size=test_size, random_state=random_state)

#     train_data = df[df['srch_id'].isin(train_queries)]
#     test_data = df[df['srch_id'].isin(test_queries)]

#     # Shuffle the data within each split
#     train_data = shuffle(train_data, random_state=random_state)
#     test_data = shuffle(test_data, random_state=random_state)

#     X_train = train_data.drop(['booking_bool','position','click_bool', 'max_rate', 'min_rate',], axis=1)
#     y_train = train_data['booking_bool']
#     X_test = test_data.drop(['booking_bool','position','click_bool', 'max_rate', 'min_rate',], axis=1)
#     y_test = test_data['booking_bool']


#     return X_train, X_test, y_train, y_test

# # Split the data into train and test sets
# X_train, X_test, y_train, y_test = split_data_with_queries(df_train2, test_size=0.2, random_state=42)

X_train = X_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]

df_test1 = df_test1.iloc[:, 1:]


# Get the column order of df_train2
column_order = X_train.columns.tolist()

# Reorder the columns of df_test1 to match the column order of df_train2
df_test1 = df_test1[column_order]
# df_test1=df_test1.drop(['random_bool', 'srch_query_affinity_score'])
              
from tqdm import tqdm
# Hyperparameter optimization
from sklearn.metrics import f1_score


def objective(trial):
    """Define the objective function"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }

    # Fit the model
    optuna_model = XGBClassifier(**params)
    optuna_model.fit(X_train, y_train)

    # Make predictions
    y_pred = optuna_model.predict(X_test)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)

    return f1



study = optuna.create_study(direction='maximize')
# Create a progress bar using tqdm
progress_bar = tqdm(total=100)

# Define the callback function to update the progress bar
def update_progress_bar(study, trial):
    progress_bar.update(1)
    
study.optimize(objective, n_trials=100, callbacks=[update_progress_bar])
progress_bar.close()









# Get the best trial and its parameters
best_trial = study.best_trial
best_params = best_trial.params

# Train the final model with the best parameters
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Make predictions on df_test1 (without target variable)

#prediction_probabilities = model.predict_proba(df_test1.drop(['random_bool', 'srch_query_affinity_score'], axis=1))[:, 1]

prediction_probabilities = model.predict_proba(df_test1)[:, 1]

rankings_df = pd.DataFrame(columns=['srch_id', 'prop_id', 'Ranking', 'BookingProbability'])
ranking = np.argsort(-prediction_probabilities) + 1

rankings_df['srch_id'] = df_test1['srch_id']  # Assuming you have a 'SearchId' column in df_test1
rankings_df['prop_id'] = df_test1['prop_id']  # Assuming you have a 'PropertyId' column in df_test1
rankings_df['Ranking'] = ranking
rankings_df['BookingProbability'] = prediction_probabilities

rankings_df=rankings_df.sort_values(['srch_id', 'BookingProbability'],ascending = [True, False])

# rankings_df.reset_index(drop=True, inplace=True)

# Create a new column to store the unique search ID rankings
rankings_df['SearchIdRanking'] = rankings_df.groupby('srch_id').cumcount() + 1

# Reset the index of the DataFrame
rankings_df.reset_index(drop=True, inplace=True)

# Print the rankings DataFrame
#print(rankings_df)
df=pd.DataFrame()
new_df = pd.DataFrame()
new_df = rankings_df[['srch_id', 'prop_id']].copy()
df = new_df[['srch_id', 'prop_id']]
df['srch_id'] = df['srch_id'].astype('int')
df['prop_id'] = df['prop_id'].astype('int')





df.to_csv('C:/Users/charl/OneDrive/Documents/VU vakken/DMT/vu-dmt-assigment-2-2023/kaggle4.csv',index=False)



df = df.drop_duplicates()



for i in df_test1.columns:
    print(i)


#-----------------------------------------------------------------------------------------------









































import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import ndcg_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import ndcg_score
from xgboost import XGBClassifier
import optuna
import numpy as np

df_test1=df_test.iloc[:10000]
df_train2=df_train.iloc[:10000]

# df_test1=df_test
# df_train2=df_train
df_test1 = df_test1.iloc[:, 2:]
df_train2 = df_train2.iloc[:, 3:]

df_train_unique = df_train[['prop_country_id', 'prop_id','booked_rank']].drop_duplicates()

df_test1 = df_test1.merge(df_train_unique, on=['prop_country_id', 'prop_id'], how='left')
 

# Get the column order of df_train2
column_order = df_train2.columns.tolist()

# Reorder the columns of df_test1 to match the column order of df_train2
df_test1 = df_test1[column_order]


# Get the column names of df1 and df2
df1_columns = set(df_test1.columns)
df2_columns = set(df_train2.columns)

# Columns present in df1 but not in df2
columns_only_in_df1 = df1_columns - df2_columns

# Columns present in df2 but not in df1
columns_only_in_df2 = df2_columns - df1_columns

# Print the results
print("Columns only in df1:", columns_only_in_df1)
print("Columns only in df2:", columns_only_in_df2)

from sklearn.metrics import ndcg_score
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd
import optuna

# Hyperparameter optimization
def objective(trial):
    """Define the objective function"""
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'ndcg@5',
        'use_label_encoder': False
    }

    ndcg_scores = []

    for query_id in df_train2['srch_id'].unique():
        query_mask = df_train2['srch_id'] == query_id
        X_train_query = df_train2.loc[query_mask].drop(['srch_id', 'target'], axis=1)
        y_train_query = df_train2.loc[query_mask]['target']
        group_sizes = [len(y_train_query)]

        # Fit the model
        optuna_model = XGBRanker(**params)
        optuna_model.fit(X_train_query, y_train_query, group_sizes)

        # Make predictions on the training set
        y_pred = optuna_model.predict(X_train_query)

        # Calculate NDCG@5 score
        ndcg = ndcg_score([y_train_query], [y_pred], k=5)
        ndcg_scores.append(ndcg)

    # Calculate average NDCG@5 score across queries
    avg_ndcg = np.mean(ndcg_scores)

    return avg_ndcg

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_score = study.best_value



# Train the final model with the best parameters
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)




# Make predictions on df_test1 (without target variable)

#prediction_probabilities = model.predict_proba(df_test1.drop(['random_bool', 'srch_query_affinity_score'], axis=1))[:, 1]

prediction_probabilities = model.predict_proba(df_test1)[:, 1]

rankings_df = pd.DataFrame(columns=['srch_id', 'prop_id', 'Ranking', 'BookingProbability'])
ranking = np.argsort(-prediction_probabilities) + 1

rankings_df['srch_id'] = df_test1['srch_id']  # Assuming you have a 'SearchId' column in df_test1
rankings_df['prop_id'] = df_test1['prop_id']  # Assuming you have a 'PropertyId' column in df_test1
rankings_df['Ranking'] = ranking
rankings_df['BookingProbability'] = prediction_probabilities

rankings_df=rankings_df.sort_values(['srch_id', 'BookingProbability'],ascending = [True, False])

# rankings_df.reset_index(drop=True, inplace=True)

# Create a new column to store the unique search ID rankings
rankings_df['SearchIdRanking'] = rankings_df.groupby('srch_id').cumcount() + 1

# Reset the index of the DataFrame
rankings_df.reset_index(drop=True, inplace=True)

# Print the rankings DataFrame
#print(rankings_df)
df=pd.DataFrame()
new_df = pd.DataFrame()
new_df = rankings_df[['srch_id', 'prop_id']].copy()
df = new_df[['srch_id', 'prop_id']]
df['srch_id'] = df['srch_id'].astype('int')
df['prop_id'] = df['prop_id'].astype('int')

















# Count the number of missing values in each column
missing_values = rankings_df.isna().sum()

# Calculate the percentage of missing values in each column
percent_missing = (missing_values / len(rankings_df)) * 100
 











# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# model = XGBClassifier(use_label_encoder=False, 
#                       eval_metric='mlogloss')
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(classification_report(y_test, y_pred))

# def objective(trial):
#     """Define the objective function"""

#     params = {
#         'max_depth': trial.suggest_int('max_depth', 1, 9),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
#         'n_estimators': trial.suggest_int('n_estimators', 50, 500),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
#         'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
#         'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
#         'eval_metric': 'mlogloss',
#         'use_label_encoder': False
#     }

#     # Fit the model
#     optuna_model = XGBClassifier(**params)
#     optuna_model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = optuna_model.predict(X_test)

#     # Evaluate predictions
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# print('Number of finished trials: {}'.format(len(study.trials)))
# print('Best trial:')
# trial = study.best_trial

# print('  Value: {}'.format(trial.value))
# print('  Params: ')

# for key, value in trial.params.items():
#     print('    {}: {}'.format(key, value))
    
# params = trial.params
# model = XGBClassifier(**params)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy after tuning: %.2f%%" % (accuracy * 100.0))

# print(classification_report(y_test, y_pred))

# prediction_probabilities = model.predict_proba(X_test)[:, 1]
# rankings_df = pd.DataFrame(columns=['SearchId', 'PropertyId', 'Ranking', 'BookingProbability'])
# ranking = np.argsort(-prediction_probabilities) + 1

# rankings_df['SearchId'] = X_test['srch_id']  # Assuming you have a 'SearchId' column in X_test
# rankings_df['PropertyId'] = X_test['prop_id']  # Assuming you have a 'PropertyId' column in X_test
# rankings_df['Ranking'] = ranking
# rankings_df['BookingProbability'] = prediction_probabilities

# rankings_df.reset_index(drop=True, inplace=True)

# rankings_df.sort_values(['SearchId', 'Ranking'], inplace=True)
# rankings_df.reset_index(drop=True, inplace=True)
# # Sort the DataFrame by SearchId and BookingProbability
# rankings_df.sort_values(['SearchId', 'BookingProbability'], inplace=True)

# # Create a new column to store the unique search ID rankings
# rankings_df['SearchIdRanking'] = rankings_df.groupby('SearchId').cumcount() + 1

# # Reset the index of the DataFrame
# rankings_df.reset_index(drop=True, inplace=True)

#------------------------------------------------------------------------------------------------------

# Step 1: Prepare the dataset






for i in df_test1.columns:
    print(i)




# train_columns = set(X_train.columns)
# test_columns = set(df_test1.columns)

# # Columns present in X_train but missing in df_test1
# train_not_test = train_columns - test_columns

# # Columns present in df_test1 but missing in X_train
# test_not_train = test_columns - train_columns

# print("Columns in X_train but missing in df_test1:", train_not_test)
# print("Columns in df_test1 but missing in X_train:", test_not_train)







# #For the kaggle

# import optuna

# # Step 1: Prepare the dataset
# X_train = df_train2.drop(['booking_bool', 'random_bool', 'click_bool', 'position'], axis=1)
# y_train = df_train2['booking_bool']
# X_test = df_test1

# expected_feature_names = [
#     'Unnamed: 0', 'srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
#     'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_log_historical_price', 'price_usd',
#     'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
#     'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
#     'orig_destination_distance', 'random_bool', 'rating_relto_region', 'rating_relto_search',
#     'review_relto_search', 'review_relto_region', 'prop_location_score', 'price_per_night', 'cluster_label',
#     'booked_rank', 'booked_percent', 'click_percent', 'total_booked', 'total_clicked', 'search_count'
# ]

# X_train = X_train.reindex(columns=expected_feature_names)


# # Hyperparameter optimization
# def objective(trial):
#     """Define the objective function"""

#     params = {
#         'max_depth': trial.suggest_int('max_depth', 1, 9),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
#         'n_estimators': trial.suggest_int('n_estimators', 50, 500),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
#         'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
#         'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
#         'eval_metric': 'mlogloss',
#         'use_label_encoder': False
#     }

#     # Fit the model
#     optuna_model = XGBClassifier(**params)
#     optuna_model.fit(X_train, y_train)

#     # Make predictions on the validation set
#     y_pred = optuna_model.predict(X_test)

#     # Evaluate predictions
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# # Get the best trial and its parameters
# best_trial = study.best_trial
# best_params = best_trial.params

# # Train the final model with the best parameters
# model = XGBClassifier(**best_params)
# model.fit(X_train, y_train)







# X_test = df_test1




# # Make predictions on df_test1 (without target variable)
# prediction_probabilities = model.predict_proba(X_test)[:, 1]
# rankings_df = pd.DataFrame(columns=['SearchId', 'PropertyId', 'Ranking', 'BookingProbability'])
# ranking = np.argsort(-prediction_probabilities) + 1

# rankings_df['SearchId'] = df_test1['SearchId']  # Assuming you have a 'SearchId' column in df_test1
# rankings_df['PropertyId'] = df_test1['PropertyId']  # Assuming you have a 'PropertyId' column in df_test1
# rankings_df['Ranking'] = ranking
# rankings_df['BookingProbability'] = prediction_probabilities

# rankings_df.sort_values(['SearchId', 'Ranking'], inplace=True)
# rankings_df.reset_index(drop=True, inplace=True)

# # Create a new column to store the unique search ID rankings
# rankings_df['SearchIdRanking'] = rankings_df.groupby('SearchId').cumcount() + 1

# # Reset the index of the DataFrame
# rankings_df.reset_index(drop=True, inplace=True)

# # Print the rankings DataFrame
# print(rankings_df)

