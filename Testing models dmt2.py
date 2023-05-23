# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:41:12 2023

@author: charl
"""


import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('C:/Users/charl/OneDrive/Documents/VU vakken/DMT/vu-dmt-assigment-2-2023/training_set_VU_DM.csv', sep=',', header=0)
data=df.iloc[:3000]


#AGNES GEDEELTE
# transform integers to strings for categorical data
data['srch_destination_id'] = data['srch_destination_id'].astype(str)
data['site_id'] = data['site_id'].astype(str)
data['visitor_location_country_id'] = data['visitor_location_country_id'].astype(str)
data['prop_country_id'] = data['prop_country_id'].astype(str)
data['prop_id'] = data['prop_id'].astype(str)

# transform integers to boleans
data["prop_brand_bool"] = data["prop_brand_bool"].astype(bool)
data["srch_saturday_night_bool"] = data["srch_saturday_night_bool"].astype(bool)
data["random_bool"] = data["random_bool"].astype(bool)
data["promotion_flag"] = data["promotion_flag"].astype(bool)

# transform all integers to floats to compute means and z-scores
for col in data.columns:
    if data[col].dtype == 'int64':
        data[col] = data[col].astype(float) 



# add feature that calculates how much the hotel rating is above the average rating of the hotels in the same region 
data['rating_relto_region'] = data.groupby('srch_destination_id')['prop_starrating'].transform(lambda x: x - x.mean())

# add feature that indicates how much the hotels rating is in same price range
data['rating_relto_search'] = data.groupby('srch_id')['prop_starrating'].transform(lambda x: x - x.mean())


# todo check if review score is above average of other search results
data['review_relto_search'] = data.groupby('srch_id')['prop_review_score'].transform(lambda x: x - x.mean())
data['review_relto_region'] = data.groupby('srch_destination_id')['prop_review_score'].transform(lambda x: x - x.mean())

# compute the mean of the two columns prop_location_score1 and prop_location_score2 
data['prop_location_score2'] = data['prop_location_score2'].fillna(data['prop_location_score1'])
data['prop_location_score1'] = data['prop_location_score1'].fillna(data['prop_location_score2'])
data['prop_location_score'] = data[['prop_location_score2', 'prop_location_score1']].mean(axis=1)
data = data.drop(['prop_location_score1', 'prop_location_score2'], axis=1)




df_everything=data


#df_everything=df
df_everything['price_per_night'] = df_everything['price_usd'] / df_everything['srch_length_of_stay'] 

#---------------------------------------------------------------------------------------------------------------
# This part is to deal with al the comp columns. 
df_everything['comp1_rate'] =df_everything['comp1_rate'] * df_everything['comp1_rate_percent_diff']
df_everything['comp2_rate'] =df_everything['comp2_rate'] * df_everything['comp2_rate_percent_diff']
df_everything['comp3_rate'] =df_everything['comp3_rate'] * df_everything['comp3_rate_percent_diff']
df_everything['comp4_rate'] =df_everything['comp4_rate'] * df_everything['comp4_rate_percent_diff']
df_everything['comp5_rate'] =df_everything['comp5_rate'] * df_everything['comp5_rate_percent_diff']
df_everything['comp6_rate'] =df_everything['comp6_rate'] * df_everything['comp6_rate_percent_diff']
df_everything['comp7_rate'] =df_everything['comp7_rate'] * df_everything['comp7_rate_percent_diff']
df_everything['comp8_rate'] =df_everything['comp8_rate'] * df_everything['comp8_rate_percent_diff']

rate_cols = [col for col in df_everything.columns if col.endswith('_rate')]
percent_diff_cols = [col for col in df_everything.columns if col.endswith('_percent_diff')]
avail_col = [col for col in df_everything.columns if col.endswith('_inv')]

cols_to_update = {
    'comp1_rate':'comp1_inv',
    'comp2_rate':'comp2_inv',
    'comp3_rate':'comp3_inv',
    'comp4_rate':'comp4_inv',
    'comp5_rate':'comp5_inv',
    'comp6_rate':'comp6_inv',
    'comp7_rate':'comp7_inv',
    'comp8_rate':'comp8_inv'
    
   
    # add more columns to update here
}
# loop through the columns to update and set them to NaN, only the ones where the comp hotel is NOT available. 
for col, condition_col in cols_to_update.items():
    df_everything[col] = np.where(df_everything[condition_col].isin([1, np.nan]), np.nan, df_everything[col])
    
  
df_everything["max_rate"] = df_everything[rate_cols].max(axis=1)
df_everything["min_rate"] = df_everything[rate_cols].min(axis=1)

df_everything['max_rate'] = np.where(df_everything['max_rate'] < 0, np.nan, df_everything['max_rate'])
df_everything['min_rate'] = np.where(df_everything['min_rate'] > 0, np.nan, df_everything['min_rate'])      
#dropping columns
cols_to_drop = rate_cols + percent_diff_cols + avail_col
df_everything.drop(cols_to_drop, axis=1, inplace=True)

#-----------------------------------------------------------------------------------------------------------------
#TO DO column toevoegen met ratio hoevaak gelikt dit hotel, ratio hoe vaak geboekt dit hotel based on Location ID



# Perform groupby on hotel ID and calculate aggregations
df_grouped1_hotel = df_everything.groupby(['prop_country_id', 'prop_id'] ).agg(
    search_count=('prop_id', 'count'),
    total_clicked = ('click_bool','sum'),
    total_booked = ('booking_bool','sum')
    )

df_grouped1_hotel['click_percent'] = df_grouped1_hotel['total_clicked'] / df_grouped1_hotel['search_count'] * 100
df_grouped1_hotel['booked_percent'] = df_grouped1_hotel['total_booked'] / df_grouped1_hotel['search_count'] * 100



# Rank hotels within each prop_country_id based on booked_percent
df_grouped1_hotel['booked_rank'] = df_grouped1_hotel.groupby('prop_country_id')['booked_percent'].rank(method='dense', ascending=False)


df_everything = df_everything.merge(df_grouped1_hotel[['booked_rank','booked_percent','click_percent','total_booked','total_clicked','search_count']], 
                                on=['prop_country_id', 'prop_id'], how='left')

df_everything = df_everything.drop(['date_time','visitor_hist_starrating','visitor_hist_adr_usd'], axis=1)

# Calculate the percentage of NaN values in each column
nan_percentage = df_everything.isnull().mean()

# Get the column names that have more than 70% NaN values
columns_to_drop = nan_percentage[nan_percentage > 0.85].index

# Drop the columns from the DataFrame
df_everything = df_everything.drop(columns_to_drop, axis=1)


# Assuming your dataframe is named 'df_cleaned'
#df_everything = df_everything.fillna(df_everything.mean())

#-----------------------------------------------------------------------------------------------------------------

# #MISSING VALUES
# data = df_everything['orig_destination_distance']
# data= data.dropna()
# # Perform Shapiro-Wilk test
# statistic, p_value = stats.shapiro(data)

# # Print the test result
# print("Shapiro-Wilk test statistic:", statistic)
# print("p-value:", p_value)
# print("The data is normally distributed." if p_value > 0.05 else "The data is not normally distributed.")

#FILLING IN  MISSING VALUES-------------------------------------------------------------------------------
#FIRST DESTINATION DISTANCE---------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from fancyimpute import IterativeImputer

# Assume your data is in a pandas DataFrame called 'df_everything'
# Select the columns with missing values
columns_with_missing = ['orig_destination_distance']

# Create a copy of the data to preserve the original
df_imputed = df_everything.copy()

# Group the data by prop_country_id and visitor_location_country_id
groups = df_imputed.groupby(['prop_country_id', 'visitor_location_country_id'])

# Iterate over the groups
for group, group_data in groups:
    # Extract the group-specific data
    X = group_data.drop(columns_with_missing, axis=1)
    y = group_data[columns_with_missing]

    # Perform MICE imputation within the group
    mice_imputer = IterativeImputer()
    X_imputed = mice_imputer.fit_transform(X, y)

    # Update the imputed values in the original DataFrame
    df_imputed.loc[group_data.index, columns_with_missing] = X_imputed[:, 0]  # Update only the first column



# The missing values in 'df_imputed' have been imputed with MICE using group-specific imputation

print(df_imputed['orig_destination_distance'].value_counts())
print(df_everything['orig_destination_distance'].value_counts())

# Calculate means and standard deviations for orig_destination_distance
original_mean_distance = df_everything['orig_destination_distance'].mean()
imputed_mean_distance = df_imputed['orig_destination_distance'].mean()
original_std_distance = df_everything['orig_destination_distance'].std()
imputed_std_distance = df_imputed['orig_destination_distance'].std()


# Assign the imputed values to the original DataFrame


df_everything['orig_destination_distance'].describe()

from fancyimpute import IterativeImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Assume your data is in a pandas DataFrame called 'df'
target_variable = 'orig_destination_distance'

# Drop rows with missing values in the target variable
df_cleaned = df_everything.dropna(subset=[target_variable])

# Get the indices of the dropped rows
dropped_indices = df_everything.index.difference(df_cleaned.index)

# Drop the same rows from df_imputed
df_imputed_cleaned = df_imputed.drop(dropped_indices)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_imputed_cleaned.drop(target_variable, axis=1),
                                                    df_cleaned[target_variable],  # Use original values here
                                                    test_size=0.2,
                                                    random_state=42)

# Train a model on the imputed data
model = HistGradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)


import matplotlib.pyplot as plt

# Scatter plot comparing imputed and observed values
plt.scatter(df_everything['orig_destination_distance'], df_imputed['orig_destination_distance'])
plt.xlabel('Observed Values')
plt.ylabel('Imputed Values')
plt.title('Imputed vs. Observed Values')
plt.show()

# Density plot comparing imputed and observed values
plt.hist(df_everything['orig_destination_distance'], bins=30, alpha=1, label='Observed')
plt.hist(df_imputed['orig_destination_distance'], bins=30, alpha=0.5, label='Imputed')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Imputed vs. Observed Values')
plt.legend()
plt.show()


print(len(df_everything['orig_destination_distance']))
print(len(df_imputed['orig_destination_distance']))

df_everything['orig_destination_distance'] = df_imputed['orig_destination_distance']



#FILLING IN MAX AND MIN RATE-------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from fancyimpute import IterativeImputer

# Assume your data is in a pandas DataFrame called 'df_everything'
# Select the columns with missing values
columns_with_missing = ['max_rate']

# Create a copy of the data to preserve the original
df_imputed = df_everything.copy()

# Group the data by prop_country_id and visitor_location_country_id
groups = df_imputed.groupby(['prop_id'])

# Iterate over the groups
for group, group_data in groups:
    # Extract the group-specific data
    X = group_data.drop(columns_with_missing, axis=1)
    y = group_data[columns_with_missing]

    # Perform MICE imputation within the group
    mice_imputer = IterativeImputer()
    X_imputed = mice_imputer.fit_transform(X, y)

    # Update the imputed values in the original DataFrame
    df_imputed.loc[group_data.index, columns_with_missing] = X_imputed[:, 0]  # Update only the first column

# The missing values in 'df_imputed' have been imputed with MICE using group-specific imputation




# Calculate means and standard deviations for orig_destination_distance
original_mean_distance = df_everything['max_rate'].mean()
imputed_mean_distance = df_imputed['max_rate'].mean()
original_std_distance = df_everything['max_rate'].std()
imputed_std_distance = df_imputed['max_rate'].std()



from fancyimpute import IterativeImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Assign the imputed values to the original DataFrame

# Assume your data is in a pandas DataFrame called 'df'
target_variable = 'max_rate'

# Drop rows with missing values in the target variable
df_cleaned = df_everything.dropna(subset=[target_variable])

# Get the indices of the dropped rows
dropped_indices = df_everything.index.difference(df_cleaned.index)

# Drop the same rows from df_imputed
df_imputed_cleaned = df_imputed.drop(dropped_indices)

# Spli
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_imputed_cleaned.drop(target_variable, axis=1),
                                                    df_cleaned[target_variable],  # Use original values here
                                                    test_size=0.2,
                                                    random_state=42)

# Train a model on the imputed data
model = HistGradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

df_everything['max_rate'].describe()



import matplotlib.pyplot as plt

# Scatter plot comparing imputed and observed values
plt.scatter(df_everything['max_rate'], df_imputed['max_rate'])
plt.xlabel('Observed Values')
plt.ylabel('Imputed Values')
plt.title('Imputed vs. Observed Values')
plt.show()

# Density plot comparing imputed and observed values
plt.hist(df_everything['max_rate'], bins=30, alpha=1, label='Observed')
plt.hist(df_imputed['max_rate'], bins=30, alpha=0.5, label='Imputed')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Imputed vs. Observed Values')
plt.legend()
plt.show()


print(len(df_everything['max_rate']))
print(len(df_imputed['max_rate']))

df_everything['max_rate'] = df_imputed['max_rate']

#MIN -------------------------------------
# Assume your data is in a pandas DataFrame called 'df_everything'
# Select the columns with missing values
columns_with_missing = ['min_rate']

# Create a copy of the data to preserve the original
df_imputed = df_everything.copy()

# Group the data by prop_country_id and visitor_location_country_id
groups = df_imputed.groupby(['prop_id'])

# Iterate over the groups
for group, group_data in groups:
    # Extract the group-specific data
    X = group_data.drop(columns_with_missing, axis=1)
    y = group_data[columns_with_missing]

    # Perform MICE imputation within the group
    mice_imputer = IterativeImputer()
    X_imputed = mice_imputer.fit_transform(X, y)

    # Update the imputed values in the original DataFrame
    df_imputed.loc[group_data.index, columns_with_missing] = X_imputed[:, 0]  # Update only the first column

# The missing values in 'df_imputed' have been imputed with MICE using group-specific imputation




# Calculate means and standard deviations for orig_destination_distance
original_mean_distance = df_everything['min_rate'].mean()
imputed_mean_distance = df_imputed['min_rate'].mean()
original_std_distance = df_everything['min_rate'].std()
imputed_std_distance = df_imputed['min_rate'].std()



from fancyimpute import IterativeImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Assign the imputed values to the original DataFrame

# Assume your data is in a pandas DataFrame called 'df'
target_variable = 'min_rate'

# Drop rows with missing values in the target variable
df_cleaned = df_everything.dropna(subset=[target_variable])

# Get the indices of the dropped rows
dropped_indices = df_everything.index.difference(df_cleaned.index)

# Drop the same rows from df_imputed
df_imputed_cleaned = df_imputed.drop(dropped_indices)

# Spli
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_imputed_cleaned.drop(target_variable, axis=1),
                                                    df_cleaned[target_variable],  # Use original values here
                                                    test_size=0.2,
                                                    random_state=42)

# Train a model on the imputed data
model = HistGradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

df_everything['min_rate'].describe()

# Scatter plot comparing imputed and observed values
plt.scatter(df_everything['min_rate'], df_imputed['min_rate'])
plt.xlabel('Observed Values')
plt.ylabel('Imputed Values')
plt.title('Imputed vs. Observed Values')
plt.show()

# Density plot comparing imputed and observed values
plt.hist(df_everything['min_rate'], bins=30, alpha=1, label='Observed')
plt.hist(df_imputed['min_rate'], bins=30, alpha=0.5, label='Imputed')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Imputed vs. Observed Values')
plt.legend()
plt.show()


print(len(df_everything['min_rate']))
print(len(df_imputed['min_rate']))

df_everything['min_rate'] = df_imputed['min_rate']


# MISSING VALUE REVIEW SCORE:------------------------------

#MIN -------------------------------------
# Assume your data is in a pandas DataFrame called 'df_everything'
# Select the columns with missing values

columns_with_missing = ['prop_review_score']

# Create a copy of the data to preserve the original
df_imputed = df_everything.copy()

# Perform MICE imputation
imputer = IterativeImputer()
df_imputed[columns_with_missing] = imputer.fit_transform(df_imputed[columns_with_missing])

# The missing values in 'df_imputed' have been imputed with MICE using group-specific imputation




# Calculate means and standard deviations for orig_destination_distance
original_mean_distance = df_everything['prop_review_score'].mean()
imputed_mean_distance = df_imputed['prop_review_score'].mean()
original_std_distance = df_everything['prop_review_score'].std()
imputed_std_distance = df_imputed['prop_review_score'].std()



from fancyimpute import IterativeImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Assign the imputed values to the original DataFrame

# Assume your data is in a pandas DataFrame called 'df'
target_variable = 'prop_review_score'

# Drop rows with missing values in the target variable
df_cleaned = df_everything.dropna(subset=[target_variable])

# Get the indices of the dropped rows
dropped_indices = df_everything.index.difference(df_cleaned.index)

# Drop the same rows from df_imputed
df_imputed_cleaned = df_imputed.drop(dropped_indices)

# Spli
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_imputed_cleaned.drop(target_variable, axis=1),
                                                    df_cleaned[target_variable],  # Use original values here
                                                    test_size=0.2,
                                                    random_state=42)

# Train a model on the imputed data
model = HistGradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

df_everything['prop_review_score'].describe()

df_everything['prop_review_score'] = df_imputed['prop_review_score']


import matplotlib.pyplot as plt

# Scatter plot comparing imputed and observed values
plt.scatter(df_everything['prop_review_score'], df_imputed['prop_review_score'])
plt.xlabel('Observed Values')
plt.ylabel('Imputed Values')
plt.title('Imputed vs. Observed Values')
plt.show()

# Density plot comparing imputed and observed values
plt.hist(df_everything['prop_review_score'], bins=30, alpha=1, label='Observed')
plt.hist(df_imputed['prop_review_score'], bins=30, alpha=0.5, label='Imputed')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Imputed vs. Observed Values')
plt.legend()
plt.show()


print(len(df_everything['prop_review_score']))
print(len(df_imputed['prop_review_score']))


import matplotlib.pyplot as plt



df_everything['prop_review_score'] = df_imputed['prop_review_score']





















#KMEANS AS A FEATURE --------------------------------------------------------------------------


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Assuming 'df_merged' is your DataFrame with missing values
X = df_everything.values
# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Deciding aount of clusters----------------------
wcss = []
max_clusters = 8

for n_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_imputed)
    wcss.append(kmeans.inertia_)

# Plot the within-cluster sum of squares (WCSS) against the number of clusters
plt.plot(range(1, max_clusters + 1), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()


n_clusters = 3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



desired_variances = [0.80, 0.85, 0.90, 0.95, 0.99]
silhouette_scores = []
db_indices = []
ch_indices = []

for desired_variance in desired_variances:
    # Perform PCA with the desired variance
    pca = PCA(n_components=desired_variance)
    X_pca = pca.fit_transform(X_imputed)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_pca)

    # Calculate performance metrics
    cluster_labels = kmeans.labels_
    silhouette = silhouette_score(X_pca, cluster_labels)
    db_index = davies_bouldin_score(X_pca, cluster_labels)
    ch_index = calinski_harabasz_score(X_pca, cluster_labels)

    # Append scores to lists
    silhouette_scores.append(silhouette)
    db_indices.append(db_index)
    ch_indices.append(ch_index)

# Plot the performance metrics against the desired variances
plt.plot(desired_variances, silhouette_scores, label='Silhouette')
plt.plot(desired_variances, db_indices, label='Davies-Bouldin')
plt.plot(desired_variances, ch_indices, label='Calinski-Harabasz')
plt.xlabel('Desired Variance Explained')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()


# Perform dimensionality reduction using PCA
pca = PCA()
X_pca = pca.fit_transform(X_imputed)

# Calculate the cumulative explained variance ratio
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Find the optimal number of components that explain a desired amount of variance
desired_variance = 0.95  # Set your desired amount of variance explained
n_components = np.argmax(explained_variance_ratio >= desired_variance) + 1

# Use the determined number of components for dimensionality reduction
pca_final = PCA(n_components=n_components)
X_final = pca_final.fit_transform(X_imputed)

# Specify the number of clusters (K)
n_clusters = 3

# Initialize and fit the K-means model
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_final)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Access the cluster centers
cluster_centers = kmeans.cluster_centers_

# Add the cluster labels as a new column in the DataFrame
df_imputed['cluster_label'] = cluster_labels

# Evaluate the clustering results using silhouette score
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(X_final, cluster_labels)
print("Silhouette score:", silhouette)


#TRYING MODELS


# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# import numpy as np
# import pandas as pd

# # Step 1: Prepare the dataset
# X = df_everything.drop(['booking_bool', 'random_bool', 'click_bool'], axis=1)
# y = df_everything['booking_bool']

# # Apply PCA for dimensionality reduction
# pca = PCA()
# X_pca = pca.fit_transform(X)

# # Normalize the feature data
# scaler = MinMaxScaler()
# X_normalized = scaler.fit_transform(X_pca)

# # Calculate the cumulative explained variance ratio
# explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# # Find the optimal number of components that explain a desired amount of variance
# desired_variance = 0.80  # Set your desired amount of variance explained
# n_components = np.argmax(explained_variance_ratio >= desired_variance) + 1

# # Use the determined number of components for dimensionality reduction
# pca_final = PCA(n_components=n_components)
# X_final = pca_final.fit_transform(X)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# # Step 2: Train the KNN model on the whole training set
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(X_train, y_train)

# # Step 3: Predict probabilities for the test set
# test_probabilities = knn.predict_proba(X_test)[:, 1]

# # Create a new DataFrame to store the rankings information
# rankings_df = pd.DataFrame(columns=['SearchId', 'PropertyId', 'Ranking'])

# # Assign rankings to the test data
# ranking = np.argsort(-test_probabilities) + 1  # Assign the ranking based on descending probabilities
# ranked_hotel_ids = df_everything.loc[y_test.index, 'prop_id'].values

# # Add the search IDs, property IDs, and rankings to the rankings DataFrame
# rankings_df['SearchId'] = df_everything.loc[y_test.index, 'srch_id'].values
# rankings_df['PropertyId'] = ranked_hotel_ids
# rankings_df['Ranking'] = ranking

# Reset the index of the DataFrame
# rankings_df.reset_index(drop=True, inplace=True)


#KNN AS A RANKING MODEL -----------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Step 1: Prepare the dataset
X = df_everything.drop(['booking_bool', 'random_bool', 'click_bool'], axis=1)
y = df_everything['booking_bool']

# Apply PCA for dimensionality reduction
pca = PCA()
X_pca = pca.fit_transform(X)

# Normalize the feature data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_pca)

# Calculate the cumulative explained variance ratio
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Find the optimal number of components that explain a desired amount of variance
desired_variance = 0.95  # Set your desired amount of variance explained
n_components = np.argmax(explained_variance_ratio >= desired_variance) + 1

# Use the determined number of components for dimensionality reduction
pca_final = PCA(n_components=n_components)
X_final = pca_final.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Step 2: Train the KNN model on the whole training set
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Step 3: Predict probabilities for the test set
test_probabilities = knn.predict_proba(X_test)[:, 1]

# Create a new DataFrame to store the rankings information
rankings_df = pd.DataFrame(columns=['SearchId', 'PropertyId', 'Ranking', 'BookingProbability'])

# Assign rankings to the test data
ranking = np.argsort(-test_probabilities) + 1  # Assign the ranking based on descending probabilities
ranked_search_ids = df_everything.loc[y_test.index, 'srch_id'].values
ranked_property_ids = df_everything.loc[y_test.index, 'prop_id'].values

# Add the search IDs, property IDs, rankings, and booking probabilities to the rankings DataFrame
rankings_df['SearchId'] = ranked_search_ids
rankings_df['PropertyId'] = ranked_property_ids
rankings_df['Ranking'] = ranking
rankings_df['BookingProbability'] = test_probabilities

# Reset the index of the DataFrame
rankings_df.reset_index(drop=True, inplace=True)


# Sort the DataFrame by SearchId and existing rankings
rankings_df.sort_values(['SearchId', 'Ranking'], inplace=True)

# Create a new column to store the query-wise rankings
rankings_df['QueryRanking'] = rankings_df.groupby('SearchId')['Ranking'].rank(method='min')

# Reset the index of the DataFrame
rankings_df.reset_index(drop=True, inplace=True)

#-------------------------------------------------------------------------------------------------

missing_values = df_everything.isna().sum()

# Calculate the percentage of missing values in each column
percent_missing = (missing_values / len(df_everything)) * 100
 

for i in df_everything.columns:
    print(i)




