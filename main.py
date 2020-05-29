import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
import string
from string import digits, punctuation
from flask import Flask
from flask_restful import Resource, Api
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import argparse
import os
from flask import request, jsonify
import keras.backend.tensorflow_backend as tb
import joblib
import sklearn
from flask import render_template

# from socket import gethostname
import socket
app = Flask(__name__)
# ip = socket.gethostbyname(gethostname()[0])
# ipf = socket.getfqdn()
# print(ipf)
# print(ip)
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def word_lenmatizer(text):
    cleanr = re.compile('<.*?>')
    text=re.sub(cleanr, ' ', text)
    clean = text.translate(str.maketrans('', '', punctuation))
    clean = clean.translate(str.maketrans('', '', digits))
    text_tokenizer=clean.split()
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    hl_lemmatized = []
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text_tokenizer]
    hl_lemmatized.append(lemm)
    return hl_lemmatized

def LSTM_predict(text,model,tokenizer):
  w_lenmatizer=word_lenmatizer(text)
  max_token = 14
  sequences = tokenizer.texts_to_sequences(w_lenmatizer)
  X = pad_sequences(sequences, maxlen=max_token)
  if np.around(model.predict(X)[0])==1:
    print(text)
    print("=============> SPAM\n")
    return 0
  else:
    print(text)
    print("=============> HAM\n")
    return 1
  

def predict(text,model,tfidf_vector):
  w_lenmatizer=word_lenmatizer(text)
  w_lenmatizer = [" ".join(x) for x in w_lenmatizer]
  X_Tfidf = tfidf_vector.transform(w_lenmatizer)
  if model.predict(X_Tfidf)[0]==1:
    print(text)
    print("=============> SPAM\n")
    return 0
  else:
    print(text)
    print("=============> HAM\n")
    return 1

LSTM_model = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/LSTM_model.pkl')
LSTM_TOKEN = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/tokenizer_LSTM.pkl')
NB_model = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/NB_model.pkl')
SVM_model = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/SVM_model.pkl')
tfidf = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/tfidf.pkl')

@app.route('/lstm',methods = ['POST'])
def pridictLSTM():
      tb._SYMBOLIC_SCOPE.value = True
      text = str(request.get_json('DATA'))
      x = LSTM_predict(text, LSTM_model, LSTM_TOKEN)
      data = {
        'Result': x
      }
      return jsonify(data)

@app.route('/svm',methods = ['POST'])
def pridictModelSVM():
            tb._SYMBOLIC_SCOPE.value = True
            text = str(request.get_json('DATA'))
            x = predict(text, SVM_model,tfidf)
            if x is None:
                x = "NULL"
            data = {
                'Result': x
            }
            return jsonify(data)

@app.route('/np',methods = ['POST'])
def pridictModelNP():
            tb._SYMBOLIC_SCOPE.value = True
            text = str(request.get_json('DATA'))
            x = predict(text, NB_model,tfidf)
            if x is None:
                x = "NULL"
            data = {
                'Result': x
            }
            return jsonify(data)

@app.route('/',methods = ['GET'])
def home(name = None):
    return render_template('upload_file.html', name=name)

if __name__ == '__main__':
      # host iphone = 172.20.10.2
    # host='192.168.43.239',
    app.run(host = '127.0.0.1', port = '1998',debug=True, threaded=False)
