
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# imports
import os
import pandas as pd
import re
import nltk
import math
import numpy as np  
import datetime 
import re
nltk.download('stopwords')
from fuzzywuzzy import process
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
nltk.download('words')
from nltk.corpus import wordnet 
from nltk.stem import PorterStemmer
from nltk import tokenize
from operator import itemgetter
from textblob import TextBlob
from time_and_date import time_convertion, date_convertion
from autocorrect import Speller

# read data
data = pd.read_csv('data/ODI-2023.csv', sep=';', header=0)

#rename column names:
data=data.rename(columns={"What programme are you in?": 'programme',
                        "Have you taken a course on machine learning?": 'ML',
                        'Have you taken a course on information retrieval?': 'IR',
                        'Have you taken a course on statistics?': 'STATS',
                        'Have you taken a course on databases?' : 'DB',
                        'What is your gender?': 'gender',
                        'I have used ChatGPT to help me with some of my study assignments ': 'ChatGPT', 
                        'When is your birthday (date)?': 'birthdate',
                        'How many students do you estimate there are in the room?':'estimate',
                        'Did you stand up to come to your previous answer    ?':'stand_up',
                        'What is your stress level (0-100)?': 'stress_level',
                        'How many hours per week do you do sports (in whole hours)? ': 'work_out',
                        'Give a random number':'random_number',
                        'Time you went to bed Yesterday':'bedtime',
                        'What makes a good day for you (1)?': 'good_day1',
                        'What makes a good day for you (2)?': 'good_day2'})

#Fix programe column
correct_programs = ['artificial intelligence','business analytics', 'bioinformatics',"computer science", 'finance', "econometrics",
                    'computational science',"ai",'cs','cls','ba', 'quantitative risk management', "qrm", 'phd', 'information science'
                    ,'human language technology']


data['programme'] = data['programme'].apply(str.lower)
data["best_match"] = data['programme'].map(lambda x: process.extractOne(x,correct_programs)[0])
data["best_score"] = data['programme'].map(lambda x: process.extractOne(x,correct_programs)[1])

for ind in data.index:
    if data["best_score"][ind] > 80:
        data['programme'][ind] = data["best_match"][ind] 

data['programme']=data['programme'].str.replace("artificial intelligence", "ai")
data['programme']=data['programme'].str.replace("computer science", "cs")
data['programme']=data['programme'].replace(["business analytics","ba"], "ba",regex=True)
data['programme']=data['programme'].str.replace("computational science", "cls",regex=True)
data['programme']=data['programme'].str.replace("quantitative risk management", "qrm",regex=True)
data['programme'].loc[data['programme'].str.contains('|'.join(["bioinformatics"]),'programme')] = "bioinformatics"
data['programme'].loc[~(data['programme'].str.contains('|'.join(correct_programs),"programme"))] = "Overig"
data['programme'].loc[data['programme'].str.contains('|'.join(["[0-9]" , 'chamber','masters','thingy']),'programme')] = np.nan

# s=data['programme'].unique()
data.drop(["best_match", "best_score"], axis = 1, inplace=True)

#YES AND NO COLUMNS
data['statistics'] = data['STATS']
data['machinelearning'] = data['ML']
data['database'] =data['DB']
data['information retrieval']= data['IR']

data.iloc[:, 2:6] = data.iloc[:, 2:6].replace(['yes','1','mu','ja'], 1)
data.iloc[:, 2:6] = data.iloc[:, 2:6].replace(['no','0','sigma','nee'],0)
data.iloc[:, 2:6] = data.iloc[:, 2:6].replace(['unknown'],np.nan)
data['total_courses'] = data.iloc[:, 2:6].sum(axis=1)

def fix_numeric_columns(data, column_name):
    data[column_name] = data[column_name].str.replace('[A-z]+', '')
    data[column_name] = data[column_name].str.replace('?', '')
    data[column_name]=pd.to_numeric(data[column_name],errors="coerce")

fix_numeric_columns(data, 'stress_level')
fix_numeric_columns(data, 'work_out')
fix_numeric_columns(data, 'random_number')
fix_numeric_columns(data, 'estimate')

#Fix nonsense values (outside the possible range)
data['estimate'].loc[(data['estimate']<0)|(data['estimate']>1500)] =np.nan
data['stress_level'].loc[(data['stress_level']<0) | (data['stress_level']>100)] =np.nan
data['work_out'].loc[(data['work_out']<0) | (data['work_out']>80)] =np.nan

#create categorical data
data.loc[data['stress_level'].between(0, 28, 'both'), 'stress_level_cat'] = 'low stress'
data.loc[data['stress_level'].between(28, 65, 'right'), 'stress_level_cat'] = 'medium stress'
data.loc[data['stress_level'].between(65, 100, 'right'), 'stress_level_cat'] = 'high stress'

data.loc[data['work_out'].between(0, 3, 'both'), 'work_out_cat'] = 'low work out'
data.loc[data['work_out'].between(3, 7, 'right'), 'work_out_cat'] = 'moderate work out'
data.loc[data['work_out'].between(7, 14, 'right'), 'work_out_cat'] = 'high work out'

data.loc[data['estimate'].between(0, 250, 'both'), 'estimation'] = 'to low estimation'
data.loc[data['estimate'].between(250, 350, 'right'), 'estimation'] = 'good estimation'
data.loc[data['estimate'].between(350, 1500, 'right'), 'estimation'] = 'to high estimation'

# convert time column
data, time_list = time_convertion(data)

# convert birthdate column
data = date_convertion(data)

# rename relative time column to rel_bedtime 
data = data.rename(columns={'relative_time': 'rel_bedtime'})

############  Creating categoricall variable from bedtime
data['rel_bedtime'].quantile([.33, .66])

data.loc[data['rel_bedtime'].between( -2000, -10, 'both'), 'Bedtime_cat'] = 'bed early'
data.loc[data['rel_bedtime'].between(-10, 59, 'right'), 'Bedtime_cat'] = 'bed average'
data.loc[data['rel_bedtime'].between(59,2500,  'right'), 'Bedtime_cat'] = 'bed late'


######################  Textual data cleaning  ######################################
stop_words = set(stopwords.words('english'))
additions=['also','review', 'great', 'well' , 'good', 'nice', 'amazing',
           'thanks','thank','recommend','recommended','great',
           'really','much', 'more' , 'recomend','would']
stop_words=stop_words.union(additions)

def remove_alphanumerics(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    output = re.sub(r'\d+', '',text_input)
    return output.lower().strip()

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(filtered_words)

def replace_abreviations(text):
    replacers = {' dm': 'direct message',
                 'thx': 'thanks',
                 'msg': 'messsage',
                 'msgs': 'messages',
                 'plz': 'please',
                 'pls': 'please',
                 ' u ': 'you',
                 ' app': 'application',
                 ' ad': 'advertisement',
                 ' ads'  'advertisement'
                 'asap': 'as soon as possible',
                 '...': '',
                 '. . .': '',
                 ' def': 'definitely',
                 'def ': 'definitely',
                 'tv': 'television',
                 'TV': 'television',   
                 'luv': 'love',
                 'nvm': 'never mind',
                 'yup': 'yes',
                 'yep': 'yes',
                 'tbh': 'to be honest',
                 'idk': ' i dont know',
                 'IMO': 'in my opinion',
                 'btw': 'by the way',
                 'ASAP': 'as soon as possible',
                 ' ofc': 'of course',
                 'ofc ': 'of course'}
    filtered_words = [word.lower() for word in text.split() if word.lower() not in replacers.keys()]
    return " ".join(filtered_words)
    

#perform contractions
def remove_contractions(text):
    text = re.sub(r'won\'t', 'will not',text)
    text = re.sub(r'wont', 'will not',text)
    text = re.sub(r'would\'t', 'would not',text)
    text = re.sub(r'wasn\'t', 'was not',text)
    text = re.sub(r'wasnt', 'was not',text)
    text = re.sub(r'could\'t', 'could not',text)
    text = re.sub(r'\'d', 'would',text)
    text = re.sub(r'can\’t', 'can not',text)
    text = re.sub(r'cant', 'can not',text)
    text = re.sub(r'\'re', ' are', text)
    text = re.sub(r'\'text', ' itext', text)
    text = re.sub(r'\'ll', ' will', text)
    text = re.sub(r'\'t', ' not', text)
    text = re.sub(r'\'ve', ' have', text)
    text = re.sub(r'\'m', ' am', text)
    return text

#perform lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    lemmatized_tokens = [ lemmatizer.lemmatize(word) for word in text.split() ] 
    return " ".join(lemmatized_tokens)

stemmer = PorterStemmer()
def stem(text):
    stemmed_tokens = [ stemmer.stem(word) for word in text.split() ] 
    return " ".join(stemmed_tokens)

spell = Speller(lang='en')
def check_spelling(text):
    corrected_words=[spell(word) for word in text.split()]
    return " ".join(corrected_words)

data['text1']=data.good_day1.map(remove_alphanumerics)
data['text2']=data.good_day2.map(remove_alphanumerics)

data['text1']=data.text1.map(remove_stopwords)
data['text2']=data.text2.map(remove_stopwords)

data['text1']=data.text1.map(check_spelling)
data['text2']=data.text2.map(check_spelling)

data['text1']=data.text1.map(remove_contractions)
data['text2']=data.text2.map(remove_contractions)

data['text1']=data.text1.map(lemmatize)
data['text2']=data.text2.map(lemmatize)

data['text1']=data.text1.map(stem)
data['text2']=data.text2.map(stem)


####### text categorization
stop_words = set(stopwords.words('english'))
lines= [ word_tokenize(line) for line in sorted(list(set(data.text1.dropna())) + list(set(data.text2.dropna()))) ]

lWordTokensText = []
for text in lines:
    for token in text:
        lWordTokensText.append(token)
                    
#calculate total number of words and sentences in corpus
iTotalTokens=len(lWordTokensText)
iTotalLines=len(data.text1.dropna()) + len(data.text2.dropna())

# Calcualte TF score - frequency of a word in all reviews
tf_score={}
                          
for token in lWordTokensText:
    if token not in stop_words:
        if token in tf_score:
            tf_score[token]+=1
        else:
            tf_score[token]=1

tf_score.update((x, y/int(iTotalLines)) for x, y in tf_score.items())

#methtod to extract top n elements from a dictionary
def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result

def lists_overlap(a, b):
    return bool(set(a) & set(b))

def categorize_text(text):
    tokens= [word for word in str(text).split()]
    if lists_overlap(['weather' , 'sun' , 'sunni' ,  'sunshin' , 'sunlight'] ,tokens):
        return 'weather'
    elif lists_overlap(['food' ,'pizza', 'meal' , 'dinner' , 'breakfast' , 'eat','chocol', 'bread'] ,tokens):
        return 'food'
    elif lists_overlap(['coffe'] , tokens):
        return 'coffee'
    elif lists_overlap(['friend' , 'drink','peopl','famili','relationship', 'love', 'beer','meet'] , tokens):
        return 'social'
    elif lists_overlap(['gym' ,'walk', 'exercis', 'footbal' , 'sport','train', 'yoga', 'run','workout' ] , tokens):
        return 'sports'
    elif lists_overlap(['tv','movi', 'netflix','music','concert','video','fun','game', 'hobby'] , tokens):
        return 'entertainment'
    elif lists_overlap([ 'lectur' , 'hobby', 'interest', 'class', 'vacat', 'holiday' , 'travel', 'noth' ,'task','rest', 'free' , 'done' , 'work' , 'time', 'relax'] , tokens):
        return 'free time'
    elif 'sex' in tokens:
        return 'sex'
    elif 'sleep' in tokens:
        return 'sleep'
    elif lists_overlap(["mood",'stress','anxieti','worry', 'product'], tokens):
        return 'mental health'
    elif lists_overlap(['win','success','accomplish','gain','grade'],tokens):
        return 'success'
    else:
        return 'other'

data['text_category1']=data.text1.map(categorize_text)
data['text_category2']=data.text2.map(categorize_text)
data = data.drop(columns=['Tijdstempel', 'random_number', 'birthdate', 'bedtime', 'DB', 'ML', 'IR', 'STATS'])

# export data as csv
if not os.path.exists('data'):
    os.makedirs('data')
data.to_csv('data/processed_data.csv', index=False)