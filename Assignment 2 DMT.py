# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:49:08 2023

@author: charl
"""
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/charl/OneDrive/Documents/VU vakken/DMT/vu-dmt-assigment-2-2023/training_set_VU_DM.csv', sep=',', header=0)

df_general_info = df.groupby('srch_id').agg(srch_id_counts =('site_id','count'),
                                      total_clicks = ('click_bool','sum'),
                                      booking_bool = ('booking_bool','sum'),
                                      total_promotions = ('promotion_flag','sum'), 
                                      min_review=('prop_review_score',  lambda x: x[x!=0].min(skipna=True)),
                                      max_review=('prop_review_score', 'max'),
                                      min_star=('prop_starrating', 'min'),
                                      max_star=('prop_starrating', 'max'),
                                      has_zero=('prop_review_score', lambda x: 0 if any(x == 0) else 1),
                                      min_usd = ('price_usd', 'min'),
                                      max_usd = ('price_usd', 'max')
                                      )


df_booked_info = df[df['booking_bool']==1]

df_booked_info = df_booked_info.set_index('srch_id')

df_everything = df_general_info.join(df_booked_info)

del df_booked_info, df_general_info

df_everything['price_per_night'] = df_everything['price_usd'] / df_everything['srch_length_of_stay'] 
df_everything['min_price_per_night'] = df_everything['min_usd'] / df_everything['srch_length_of_stay'] 
df_everything['max_price_per_night'] = df_everything['max_usd'] / df_everything['srch_length_of_stay'] 

df_everything["difference_min_review"] = df_everything['min_review'] -df_everything['prop_review_score'] 
df_everything["difference_max_review"] = df_everything['max_review'] -df_everything['prop_review_score'] 
df_everything["difference_min_price"] = df_everything['min_price_per_night'] -df_everything['price_per_night'] 
df_everything["difference_max_price"] = df_everything['max_price_per_night'] -df_everything['price_per_night'] 
df_everything["difference_min_star"] = df_everything['min_star'] -df_everything['prop_starrating'] 
df_everything["difference_max_star"] = df_everything['max_star'] -df_everything['prop_starrating'] 

df_everything['comp1_rate'] =df_everything['comp1_rate'] * df_everything['comp1_rate_percent_diff']
df_everything['comp2_rate'] =df_everything['comp2_rate'] * df_everything['comp2_rate_percent_diff']
df_everything['comp3_rate'] =df_everything['comp3_rate'] * df_everything['comp3_rate_percent_diff']
df_everything['comp4_rate'] =df_everything['comp4_rate'] * df_everything['comp4_rate_percent_diff']
df_everything['comp5_rate'] =df_everything['comp5_rate'] * df_everything['comp5_rate_percent_diff']
df_everything['comp6_rate'] =df_everything['comp6_rate'] * df_everything['comp6_rate_percent_diff']
df_everything['comp7_rate'] =df_everything['comp7_rate'] * df_everything['comp7_rate_percent_diff']
df_everything['comp8_rate'] =df_everything['comp8_rate'] * df_everything['comp8_rate_percent_diff']

rate_cols = [col for col in df.columns if col.endswith('_rate')]
percent_diff_cols = [col for col in df.columns if col.endswith('_percent_diff')]
avail_col = [col for col in df.columns if col.endswith('_inv')]

df_everything["max_rate"] = df_everything[rate_cols].max(axis=1)
df_everything["min_rate"] = df_everything[rate_cols].min(axis=1)

df_everything['max_rate'] = np.where(df_everything['max_rate'] < 0, np.nan, df_everything['max_rate'])
df_everything['min_rate'] = np.where(df_everything['min_rate'] > 0, np.nan, df_everything['min_rate'])      
 
# create max_rate_avail column
df_everything['max_rate_avail'] = 1
df_everything.loc[df_everything['max_rate'] > 0, 'max_rate_avail'] = df_everything.loc[df_everything['max_rate'] > 0, avail_col].min(axis=1)

# create min_rate_avail column
df_everything['min_rate_avail'] = 1
df_everything.loc[df_everything['min_rate'] < 0, 'min_rate_avail'] = df_everything.loc[df_everything['min_rate'] < 0, avail_col].min(axis=1)

    
#dropping columns
cols_to_drop = rate_cols + percent_diff_cols + avail_col
df_everything.drop(cols_to_drop, axis=1, inplace=True)

df_everything[['min_rate_avail', 'max_rate_avail']] = df_everything[['min_rate_avail', 'max_rate_avail']].replace({0: 1, 1: 0})



#-------------------------------------------------------------------------------------------------------------
# Count the number of missing values in each column
missing_values = df_everything.isna().sum()

# Calculate the percentage of missing values in each column
percent_missing = (missing_values / len(df_everything)) * 100
 

for i in df_everything.columns:
    print(i)



