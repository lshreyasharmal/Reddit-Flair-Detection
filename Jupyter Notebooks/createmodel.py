
# coding: utf-8

# In[1]:


from pymongo import MongoClient 
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

try: 
    connection = MongoClient() 
    print("Connected successfully!!!") 
except:   
    print("Could not connect to MongoDB")
    
database = connection.flair_database
coll_train = database.training_data4
coll_test = database.testing_data4


# In[2]:


training_data = pd.DataFrame(list(coll_train.find()))


# In[3]:


testing_data = pd.DataFrame(list(coll_test.find()))


# In[4]:


final_features = [u'author', u'comments', u'is_original_content',
       u'is_reddit_media_domain', u'is_video', u'over_18', u'permalink',
       u'secure_media', u'selftext', u'send_replies', u'title', u'url',
       u'edited', u'num_comments', u'num_duplicates', u'subreddit_subscribers',
       u'ups', u'upvote_ratio']


# In[5]:


y_test = testing_data['link_flair_text']
y_train = training_data['link_flair_text']


# In[6]:


training_data = training_data[final_features]
testing_data = testing_data[final_features]


# In[10]:



def get_given_features_from_data(feature_list, data):
#     print feature_list[0]
    final_data = np.array([x for x in data[feature_list[0]]])
    final_data = np.atleast_2d(final_data)
    if(final_data.shape[0]==1):
        final_data = final_data.T
#     print final_data.shape
    for i in range(1,len(feature_list)):
        feature = feature_list[i]
#         print feature
        data_features = np.array([x for x in data[feature]])
        reshaped_array = np.atleast_2d(data_features)
        if(reshaped_array.shape[0]==1):
            reshaped_array = reshaped_array.T
        final_data = np.concatenate((final_data,reshaped_array),axis=1)
#     print final_data.shape
    return final_data

def get_train_test_for_feature_list(feauture_list,train,test):
    final_training_data = get_given_features_from_data(feauture_list,train)
    final_testing_data = get_given_features_from_data(feauture_list,test)
    return final_training_data,final_testing_data
    

from sklearn.neural_network import MLPClassifier
def train_and_predict1(x_train,x_test,y_train,y_test):
    clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(2),random_state=1).fit(x_train, y_train)
    y_predictions = clf.predict(x_test)
    return accuracy_score(y_test, y_predictions),clf

from sklearn.ensemble import RandomForestClassifier
def train_and_predict2(x_train,x_test,y_train,y_test):
    clf = RandomForestClassifier(n_estimators=1000, max_depth=10,random_state=1).fit(x_train, y_train)
    y_predictions = clf.predict(x_test)
    return accuracy_score(y_test, y_predictions),clf

from sklearn.svm import LinearSVC
def train_and_predict3(x_train,x_test,y_train,y_test):
    clf = LinearSVC(random_state=1, tol=1e-5,C=0.1).fit(x_train,y_train)
    y_predictions = clf.predict(x_test)
    return accuracy_score(y_test, y_predictions),clf

def get_accuracy_for_features(feature_list,train,test):
    print(feature_list)
    x_train,x_test = get_train_test_for_feature_list(feature_list,train,test)
    acc1,mlp = train_and_predict1(x_train,x_test,y_train,y_test)
    acc2,rf = train_and_predict2(x_train,x_test,y_train,y_test)
    acc3,svc = train_and_predict3(x_train,x_test,y_train,y_test)
#     print "Features: " + feature_list
    print("MLP " + str(acc1))
    print("RF " + str(acc2))
    print("SVC " + str(acc3))
    print("")
    return acc1,acc2,acc3,mlp,rf,svc


# In[11]:



acc1,acc2,acc3,mlp,rf,svc = get_accuracy_for_features(final_features,training_data,testing_data)


# In[12]:


for f in final_features:
    acc1,acc2,acc3,mlp,rf,svc = get_accuracy_for_features([f],training_data,testing_data)


# In[13]:


get_accuracy_for_features([u'author', u'comments', u'permalink',
       u'secure_media', u'selftext', u'title', u'url'],training_data,testing_data)


# In[14]:


get_accuracy_for_features([u'comments', u'selftext', u'title'],training_data,testing_data)


# In[15]:


get_accuracy_for_features([u'comments',u'title'],training_data,testing_data)


# In[16]:


get_accuracy_for_features(training_data.select_dtypes(include=[np.number]).columns,training_data,testing_data)


# In[17]:


get_accuracy_for_features([u'is_original_content', u'is_reddit_media_domain', u'is_video',
       u'over_18', u'send_replies'],training_data,testing_data)


# In[18]:


get_accuracy_for_features([u'author', u'comments', u'permalink',
       u'secure_media', u'selftext', u'title', u'url',
       u'edited', u'num_comments', u'num_duplicates', u'subreddit_subscribers',
       u'ups', u'upvote_ratio'],training_data,testing_data)


# In[19]:


get_accuracy_for_features([u'author', u'comments', u'is_original_content',
       u'is_reddit_media_domain', u'is_video', u'over_18', u'permalink',
       u'secure_media', u'selftext', u'send_replies', u'title', u'url'],training_data,testing_data)


# In[20]:



acc1,acc2,acc3,mlp,rf,svc = get_accuracy_for_features(final_features,training_data,testing_data)


# In[21]:


import joblib 
joblib.dump(mlp, 'classifier.pkl') 

