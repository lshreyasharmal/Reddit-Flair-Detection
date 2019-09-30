import numpy as np
import pandas as pd
import pickle
import praw
import re
import joblib
import string
import re
import nltk
import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from django.conf import settings 


reddit = praw.Reddit(client_id="z8UhRRiFnEVZ8Q",
                     client_secret="bV7OxwG-VKjdyKxcuUihnV1lNPg",
                     password="qazwsx123",
                     username="lshreyasharmal",
                     user_agent="bot1 user agent")
final_features = [u'author', u'comments', u'is_original_content',
       u'is_reddit_media_domain', u'is_video', u'over_18', u'permalink',
       u'secure_media', u'selftext', u'send_replies', u'title', u'url',
       u'edited', u'num_comments', u'num_duplicates', u'subreddit_subscribers',
       u'ups', u'upvote_ratio']
numerical_cols = [u'edited', u'num_comments',u'num_duplicates',u'subreddit_subscribers', u'ups', u'upvote_ratio']
flairs_mapping = {1:'AskIndia',2:'Business/Finance',3:'Food',4:'Non-Political',5:'Photography',6:'Policy/Economy',7:'Politics',8:'Scheduled',9:'Science/Technology',10:'Sports',0:'Others'}
flairs_comments = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0}
flairs_ups = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0}

def get_upvotes_vs_flairs():
    print("getting upvotes")
    for i in range(len(settings.ALL_DATA.ups)):
        flairs_ups[str(settings.ALL_DATA.link_flair_text.iloc[i])]+=settings.ALL_DATA.ups.iloc[i]
    upvotes_vs_flairs = {}
    for key in flairs_ups.keys():
        upvotes_vs_flairs[flairs_mapping[int(key)]] = flairs_ups[key]
    return upvotes_vs_flairs

def get_comments_vs_flairs():
    print("getting comments")
    for i in range(len(settings.ALL_DATA.num_comments)):
        flairs_comments[str(settings.ALL_DATA.link_flair_text.iloc[i])]+=settings.ALL_DATA.num_comments.iloc[i]
    comment_vs_flairs = {}
    for key in flairs_comments.keys():
        comment_vs_flairs[flairs_mapping[int(key)]] = flairs_comments[key]
    return comment_vs_flairs

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

def get_feature_dict(s):
    to_dict = vars(s)
    y_true = s.link_flair_text
    to_dict['upvote_ratio'] = s.upvote_ratio
    author = str(s.author).split("'")
    to_dict['author'] = author[len(author)-2]
    comments = []
    s.comments.replace_more(limit=None)
    for comment in s.comments.list():
        comments.append(comment.body)
    to_dict['comments'] = comments
    sub_dict = {field:to_dict[field] for field in final_features}
    return y_true,sub_dict


def get_flair(code):
    print("getting reddit post data using praw...")
    s = reddit.submission(code)
    y_true,sub_dict = get_feature_dict(s)
    print("preprocessing...")
    new_data = hanlde_bool_and_tokenize([sub_dict])
    new_data = pd.DataFrame.from_records(new_data)
    numerical_data = new_data[numerical_cols]
    print("min max scalar...")
    numeric_X  = settings.SCALER.transform(numerical_data)
    numeric_X = pd.DataFrame(numeric_X, index=numerical_data.index, columns=numerical_data.columns)
    X = new_data.drop(numerical_cols,axis=1)
    X_final = X.join(numeric_X)
    obj_cols = X_final.select_dtypes(include=[object]).columns
    temp_X = X_final
    print("vectorizing strings...")
    for obj_col in obj_cols:
        temp_X = get_obj_column(obj_col,temp_X,settings.FAST_TEXT_MODEL)
    print("concatenating features...")
    final_data = get_given_features_from_data(final_features,temp_X)
    print("predicting...")
    return flairs_mapping[settings.CLASSIFIER.predict(final_data)[0]]
    
def predict_flair_from_url(url_link):
    code = ""
    try:
        url_link_formatted = re.sub(r'https://www.reddit.com/','',url_link)
        url_tokens = url_link_formatted.split("/")
        if url_tokens[0]!="r" or url_tokens[1]!="india":
            return "Error: Incorrect Link"
        for i in range(len(url_tokens)):
            if(url_tokens[i]=='comments'):
                code = url_tokens[i+1]
    except:
        return "Error Occurred!"
    if(code!=""):
        return get_flair(code)
    else:
        return "post code not found"

def get_given_features_from_data(feature_list, data):
    final_data = np.array([x for x in data[feature_list[0]]])
    final_data = np.atleast_2d(final_data)
    for i in range(1, len(feature_list)):
        feature = feature_list[i]
        data_features = np.array([x for x in data[feature]])
        reshaped_array = np.atleast_2d(data_features)
        if (reshaped_array.shape[0] == 1 and reshaped_array.shape[1] == 1):
            reshaped_array = reshaped_array.T
        final_data = np.concatenate((final_data, reshaped_array), axis=1)

    return final_data