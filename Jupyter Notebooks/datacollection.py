
# coding: utf-8

# In[1]:


import praw 
from praw.models import MoreComments
import numpy as np
import pandas as pd


# In[2]:


features = ['num_comments',
            'ups',
            'is_reddit_media_domain',
            'is_robot_indexable',
            'is_video',
            'no_follow',
            'title',
            'is_original_content',
            'send_replies',
            'permalink',
            'edited',
            'upvote_ratio',
            'author',
            'selftext',
            'over_18',
            'comments',
            'subreddit_subscribers',
            'secure_media',
            'num_duplicates',
            'url',
            'distinguished',
            'link_flair_text']

flairs = ['AskIndia',
        'Business/Finance',
        'Food',
        'Non-Political',
        'Photography',
        'Policy/Economy',
        'Politics',
        'Scheduled',
        'Science/Technology',
        'Sports']

flairs_mapping = {'AskIndia':1,'Business/Finance':2,'Food':3,'Non-Political':4,'Photography':5,'Policy/Economy':6,'Politics':7,'Scheduled':8,'Science/Technology':9,'Sports':10,'Others':0}

flair_count = {
        'AskIndia':0,
        'Business/Finance':0,
        'Food':0,
        'Non-Political':0,
        'Photography':0,
        'Policy/Economy':0,
        'Politics':0,
        'Scheduled':0,
        'Science/Technology':0,
        'Sports':0,
        'Others':0
}

data = []


reddit = praw.Reddit(client_id="z8UhRRiFnEVZ8Q",
                     client_secret="bV7OxwG-VKjdyKxcuUihnV1lNPg",
                     password="qazwsx123",
                     username="lshreyasharmal",
                     user_agent="bot1 user agent")
subreddit = reddit.subreddit("india")

i = 0
for submission in subreddit.top(limit=1000000):
    done = False
    post = vars(submission)
    flair_text = ""
    if post['link_flair_text'] not in flairs:
        flair_text = "Others" 
    else:
        flair_text = post['link_flair_text']
    if flair_count[flair_text] <200:
        print("post no.: " + str(i))
        post['upvote_ratio'] = submission.upvote_ratio
        
        print("author:")
        author = str(submission.author).split("'")
        post['author'] = author[len(author)-2]
        
        print("appending comments:")

        comments = []
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
                comments.append(comment.body)
        post['comments'] = comments
        post['link_flair_text'] = flairs_mapping[flair_text]
        print("getting specific features:")
        post_features = {field:post[field] for field in features}
        
        print("flair : " + flair_text)

        data.append(post_features)
        i+=1
        
    for (k,v) in flair_count.iteritems():
        if(v>=200):
            done = True
        else:
            done = False
    if done:
        break    
#         collection.insert_one(post_features)
        print("----------------")


# In[ ]:


np.save("all_data.npy", data)

