import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.preprocessing import PowerTransformer
import pickle

# import randomoversampler
from imblearn.over_sampling import RandomOverSampler


train = True

if train:
    data = pd.read_csv('data/train_cleaned.csv')
else:
    data = pd.read_csv('data/test_cleaned.csv')
    
print(data.shape)
 
# print all numerical features 
numerical_features = [
       'prop_starrating', 'prop_review_score', 'prop_log_historical_price', 
       'price_usd', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'orig_destination_distance', 'perc_clicked',
       'perc_booked', 'target', 'rating_relto_region', 'rating_relto_search',
       'review_relto_search', 'review_relto_region', 'price_per_night',
       'price_rel_to_region', 'price_rel_to_search', 'rel_distance',
       'prop_location_score', 'comb_rate', 'comb_inv']
   
    
# INVESTIGATE OUTLIERS
for feature_name in numerical_features:

    feature = data[feature_name]
    
    upper_value = np.percentile(feature,99)
    
    # check if min value is negative 
    if np.min(feature) < 0:
        lower_value = np.percentile(feature,1)
    else:
        lower_value = 0
    
    data[feature_name] = np.where(feature > upper_value, upper_value, feature)
    data[feature_name] = np.where(feature < lower_value, lower_value, feature)
    

# INVESTIGATE FEATURE TRANSFORMATIONS
best_fits = {}  # Dictionary to store the best fit for each feature

if train:
    
    # use SMOTE for hybrid approach of oversampling and undersampling for imbalanced data
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(data[numerical_features], data['target'])
    print(X_resampled.shape)

    # Iterate over each numerical feature
    for feature in numerical_features:
        best_fit = None  # Variable to store the best fit for the current feature
        best_corr = 0  # Variable to store the highest correlation for the current feature
        
        # Try different types of relationships
        for relation_type in ['linear', 'logistic', 'exponential', 'quadratic', 'sqrt']:
            # Transform the feature based on the relation type
            if relation_type == 'linear':
                transformed_feature = X_resampled[feature]
            elif relation_type == 'logistic':
                transformed_feature = 1 / (1 + np.exp(-X_resampled[feature]))
            elif relation_type == 'exponential':
                transformed_feature = np.exp(X_resampled[feature])
            elif relation_type == 'quadratic':
                transformed_feature = X_resampled[feature] ** 2
            elif relation_type == 'sqrt':
                transformed_feature = np.sqrt(X_resampled[feature])
            
            # Calculate the correlation with the target variable
            correlation = np.abs(y_resampled.corr(transformed_feature))
            
            # Update the best fit if the correlation is higher
            if correlation > best_corr:
                best_corr = correlation
                best_fit = relation_type
        
        best_fits[feature] = best_fit  # Store the best fit for the current feature
        
    # save dict to file
    with open('best_fits.pickle', 'wb') as handle:
        pickle.dump(best_fits, handle, protocol=pickle.HIGHEST_PROTOCOL)

# open dict from file
with open('best_fits.pickle', 'rb') as handle:
    best_fits = pickle.load(handle)

# Print the best fit for each feature
for feature, fit in best_fits.items():
    print(f"{feature}: Best Fit = {fit}")
    
# TRANSFORM FEATURES
for feature, fit in best_fits.items():
    if fit == 'linear':
        data[feature] = data[feature]
    elif fit == 'logistic':
        data[feature] = 1 / (1 + np.exp(-data[feature]))
    elif fit == 'exponential':
        data[feature] = np.exp(data[feature])
    elif fit == 'quadratic':
        data[feature] = data[feature] ** 2
    elif fit == 'sqrt':
        data[feature] = np.sqrt(data[feature])
        
# print data size
print(data.shape)

# export data
if train:
    data.to_csv('data/train_transformed.csv', index=False)
else:   
    data.to_csv('data/test_transformed.csv', index=False)