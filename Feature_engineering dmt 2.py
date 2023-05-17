# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:07:00 2023

@author: charl
"""
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/charl/OneDrive/Documents/VU vakken/DMT/vu-dmt-assigment-2-2023/training_set_VU_DM.csv', sep=',', header=0)
#df_everything=df.iloc[:10000]
df_everything=df
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
