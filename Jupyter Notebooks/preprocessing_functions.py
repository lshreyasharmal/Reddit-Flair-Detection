import string
import re
import nltk
import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def convert_boolean(value):
    return (1 if value else 0)

def tokenize_(sentence):
    punctuations = set(string.punctuation)
    stop_words = set(stopwords.words('english'))
    if(sentence==None):
        return ""
    sentence = sentence.encode('ascii', 'ignore').decode('ascii')
    sentence = sentence.replace("\n","")
    sentence = sentence.lower()
    sentence = re.sub(r'\d+', '', sentence)
    sentence = re.sub(r'[^\w\s]','',sentence)
    sentence = ''.join(ch for ch in sentence if ch not in punctuations)
    tokens_ = nltk.word_tokenize(sentence)
    tokens = []
    for token in tokens_:
        if("http" in token or ".com" in token or "www" in token):
            continue
        tokens.append(token)
    final_tokens = [w for w in tokens if not w in stop_words] 
    return final_tokens

def tokenize_media(dict_obj):
    if(dict_obj==None):
        return ""
    if "oembed" in dict_obj.keys():
        if "html" in dict_obj['oembed'].keys():
            html_text = BeautifulSoup(dict_obj['oembed']['html'], "lxml").text
        else:
            return tokenize_("")
    else:
        return tokenize_("")
    return tokenize_(html_text)

def tokenize_urls(link):
    link = re.sub(r'^r/', ' ', link)
    link = re.sub(r'/', ' ', link)
    link = re.sub(r'_', ' ', link)
    link = re.sub(r'https','',link)
    link = re.sub(r'www','',link)
    link = re.sub(r'.com','',link)
    return tokenize_(link)

def buildWordVector(tokens,model):
    size=10
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model.wv[word].reshape((1, size))
        except:
            vec += np.zeros(size).reshape((1, size))
        count += 1.
    if count != 0:
        vec /= count
    return vec

def get_avg_comment_vec(comments,model):
    size = 10
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for comment in comments:
        vec += buildWordVector(comment,model)
        count += 1.
    if count != 0:
        vec /= count
    return vec

def get_obj_column(col,df,model):
    if col == "comments":
        temp = [get_avg_comment_vec(x,model) for x in df[col]]
    else:
        temp = [buildWordVector(x,model) for x in df[col]]
    temp_ = [list(np.ravel(x)) for x in temp]
    temp_df = pd.DataFrame([temp_])
    temp_df_t = temp_df.transpose()
    temp_df_t.columns = [col]
    temp_df_t.index = df.index
    df = df.drop(columns = [col])
    df = df.join(temp_df_t)
    return df

def hanlde_bool_and_tokenize(data):
    for line in data:
        for k,v in line.items():
            if isinstance(v,bool):
                line[k] = convert_boolean(v)
            if(k=="secure_media"):
                line[k] = tokenize_media(v)
            if(k=="author"):
                line[k] = tokenize_(v)
            if(k=="comments"):
                converted_comments = []
                for comment in v:
                    converted_comments.append(tokenize_(comment))
                line[k] = converted_comments
            if(k=="permalink"):
                line[k] = tokenize_urls(v)
            if(k=="url"):
                line[k] = tokenize_urls(v)
            if(k=="selftext"):
                line[k] = tokenize_(v)
            if(k=="title"):
                line[k] = tokenize_(v)
    return data

