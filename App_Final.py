# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:43:43 2020

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:31:39 2020

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:11:21 2020

@author: DELL
"""
#Text Cleaning Script



from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

replace_by_space = re.compile('[/(){}\[\]\|@,;]')
replace_symbol = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('India')
STOPWORDS.add('india')

def string_form(value):
    return str(value)

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = replace_by_space.sub(' ', text) # replace certain symbols by space in text
    text = replace_symbol.sub('', text) # delete symbols from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove STOPWORDS from text
    return text



import pickle
import gensim
import praw
from praw.models import MoreComments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



# file_name = "model/random_forest_model.bin.zip"

# Use pickle to load in the pre-trained model
model = pickle.load(open("Pickle_RL_Model.pkl", "rb"))

reddit = praw.Reddit(client_id='c5b4px4PJE70tQ',
                     client_secret='W70uu68jb1-0xV_KQrQnqthnUtQ',
                     user_agent='Flare_Project')


def prediction(url):
	submission = reddit.submission(url = url)
	data = {}
	data["title"] = str(submission.title)
	data["url"] = str(submission.url)
	data["body"] = str(submission.selftext)

	submission.comments.replace_more(limit=None)
	comment = ''
	count = 0
	for top_level_comment in submission.comments:
		comment = comment + ' ' + top_level_comment.body
		count+=1
		if(count > 10):
		 	break
		
	data["comment"] = str(comment)

	data['title'] = clean_text(str(data['title']))
	data['body'] = clean_text(str(data['body']))
	data['comment'] = clean_text(str(data['comment']))
    
	combined_features = data["title"] + data["comment"] + data["body"] + data["url"]

	return model.predict([combined_features])


import flask
import pickle
import pandas as pd
#!pip install flask 
from flask import Flask, request, jsonify


# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        text = flask.request.form['url']
        
        # Get the model's prediction
        flair = prediction(str(text))
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html', original_input={'url':str(text)}, result=flair,)

@app.route('/automated_testing', methods = ['POST'])
def getfile():
    upload_file = request.files["upload_file"]
    data = upload_file.read()
    s = data
    l = []
    l = s.splitlines()
    dicts = {}
    for e in l:
        s = str(e)
        s2 = s[2:-1]
        s3 = prediction(s2)
        dicts[s2] = str(s3)

    return jsonify(dicts)

if __name__ == '__main__':
    app.run()


