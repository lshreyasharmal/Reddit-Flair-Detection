
# coding: utf-8

# In[1]:


from pymongo import MongoClient 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from gensim.models import FastText
from preprocessing_functions import *
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# In[2]:


with open("all_data.npy","rb") as file:
    data_ = np.load(file,encoding='bytes')

# keys = [x.decode("utf-8")  for x in data_[0].keys()]
keys = [x for x in data_[0].keys()]

np_data = []
for d in data_:
    new_d = {}
    for k in keys:
        new_d[k] = d[k]
#         if(isinstance(d[k.encode("utf-8")],bytes)):
#             new_d[k]=d[k.encode("utf-8")].decode("utf-8")
#         else:
#             new_d[k]=d[k.encode("utf-8")]
    np_data.append(new_d)

np.save("final_data.npy",np.array(np_data))


# In[3]:


with open("final_data.npy","rb") as file:
    data = np.load(file)


# In[4]:


data = hanlde_bool_and_tokenize(data)


# In[5]:


all_data = pd.DataFrame.from_records(data)


# In[6]:


y = all_data.link_flair_text
X = all_data.drop('link_flair_text',axis=1)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[8]:


all_texts = [X_train.title]
all_texts.extend(X_train.selftext)
for comment in X_train.comments:
    all_texts.extend(comment)
all_texts = np.array(all_texts)
for x in all_texts[0]:
    all_texts[0] = x


# In[9]:


fastText_model = FastText(all_texts, min_count=1,size=10)


# In[10]:


import joblib 
joblib.dump(fastText_model, 'fast_text_model.pkl') 


# In[11]:


numerical_cols = [u'edited', u'num_comments',u'num_duplicates',u'subreddit_subscribers', u'ups', u'upvote_ratio']
numeric_X_train = X_train[numerical_cols]
numeric_X_test = X_test[numerical_cols]

scaler = MinMaxScaler()
numeric_X_train_minmax = scaler.fit_transform(numeric_X_train)
numeric_X_train_minmax = pd.DataFrame(numeric_X_train_minmax, index=numeric_X_train.index, columns=numeric_X_train.columns)
numeric_X_train_minmax = numeric_X_train_minmax.loc[:, numeric_X_train_minmax.std() > 0]
final_numeric_cols = numeric_X_train_minmax.columns

joblib.dump(scaler, 'scaler.pkl') 
numeric_X_test_minmax = scaler.transform(numeric_X_test)
numeric_X_test_minmax = pd.DataFrame(numeric_X_test_minmax, index=numeric_X_test.index, columns=numeric_X_test.columns)
numeric_X_test_minmax = numeric_X_test_minmax[final_numeric_cols]


# In[12]:


X_train = X_train.drop(numerical_cols,axis=1)
X_test = X_test.drop(numerical_cols,axis=1)


# In[13]:


X_test_final = X_test.join(numeric_X_test_minmax)
X_train_final = X_train.join(numeric_X_train_minmax)


# In[14]:


cols_with_same_val = []
for col in X_train_final.columns:
    if isinstance(X_train_final[col].iloc[0],np.number):
        if X_train_final[col].nunique() == 1:
            cols_with_same_val.append(col)
    else:
        if X_train_final[col].isnull().all():
            cols_with_same_val.append(col)
    i=0
    flag = True
    for x in X_train_final[col]:
        if(x!=None and x!=[] and x!=""):
            flag = False
            break
    if flag == True and col not in cols_with_same_val:
        cols_with_same_val.append(col)
    
cols_with_same_val


# In[15]:


X_train_final = X_train_final.drop(columns=cols_with_same_val)
X_test_final = X_test_final.drop(columns=cols_with_same_val)


# In[16]:


FINAL_FEATURES = X_train_final.columns


# In[17]:


FINAL_FEATURES


# In[18]:


training_data = X_train_final.join(y_train)
testing_data = X_test_final.join(y_test)


# In[19]:


obj_cols = training_data.select_dtypes(include=[object]).columns


# In[20]:


temp_train = training_data
for obj_col in obj_cols:
    print(obj_col)
    temp_train = get_obj_column(obj_col,temp_train,fastText_model)


# In[21]:


temp_test = testing_data
for obj_col in obj_cols:
    print(obj_col)
    temp_test = get_obj_column(obj_col,temp_test,fastText_model)


# In[23]:


try: 
    connection = MongoClient() 
    print("Connected successfully!!!") 
except:   
    print("Could not connect to MongoDB")
    
database = connection.flair_database
coll_train = database.training_data4
coll_test = database.testing_data4


# In[24]:


coll_train.insert_many(temp_train.to_dict('records'))


# In[25]:


coll_test.insert_many(temp_test.to_dict('records'))


# In[26]:


temp_test.shape


# In[27]:


temp_train.shape

