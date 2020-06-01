import nltk
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
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from nltk.corpus import wordnet,stopwords
import glob
import pandas as pd
from sklearn.metrics import classification_report

import socket
app = Flask(__name__)

def standardize_data(text):
      # Replace email addresses with 'email'
  re_email=re.compile('[\w\.-]+@[\w\.-]+(\.[\w]+)+')
  text=re.sub(re_email,'email',text)

  # Replace URLs with 'webaddress'
  re_url=re.compile('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
  text=re.sub(re_url,'webaddress',text)

  # Replace money symbols with 'moneysymb'
  re_moneysb=re.compile('\$')
  text=re.sub(re_moneysb,'moneysb',text)

  # Remove ufeff 
  re_moneysb=re.compile('\ufeff|\\ufeff')
  text=re.sub(re_moneysb,' ',text)

  # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
  re_phonenb=re.compile('(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
  text=re.sub(re_phonenb,'phonenb',text)

  # Replace numbers with 'numbr'
  re_number=re.compile('\d+(\.\d+)?')
  text=re.sub(re_number,' numbr ',text)

  # Remove puntuation
  text=text.translate(str.maketrans('', '', punctuation))

  # Replace whitespace between terms with a single space
  re_space=re.compile('\s+')
  text=re.sub(re_space,' ',text)

  # Remove leading and trailing whitespace
  re_space=re.compile('^\s+|\s+?$')
  text=re.sub(re_space,' ',text)

  return text

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  token=[term for term in text.split() if term not in stop_words]
  return token

def word_lenmatizer(token):
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    hl_lemmatized = []
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in token]
    hl_lemmatized.append(lemm)
    return hl_lemmatized

def vectorize_lstm(w_lenmatizer,tokenizer):
  sequences = tokenizer.texts_to_sequences(w_lenmatizer)
  X = pad_sequences(sequences, maxlen=110)
  return X


def vectorize_clasifer(w_lenmatizer, tfidf_vector):
  w_lenmatizer = [" ".join(x) for x in w_lenmatizer]
  return tfidf_vector.transform(w_lenmatizer)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def load_data(path):
  all_files = glob.glob(path + "/*.csv")
  li = []
  for filename in all_files:
      data_frame = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8')
      li.append(data_frame)

  df = pd.concat(li, axis=0, ignore_index=True)
  return df

def pre_processing(text):
      standardize = standardize_data(text)
      
      tokens = remove_stopwords(standardize)

      w_lenmatizer=word_lenmatizer(tokens)
      
      return str(standardize), str(tokens), w_lenmatizer



def LSTM_predict(text,model,tokenizer):
  text = standardize_data(text)
  token =  remove_stopwords(text)
  w_lenmatizer=word_lenmatizer(token)
  max_token = 110
  sequences = tokenizer.texts_to_sequences(w_lenmatizer)
  X = pad_sequences(sequences, maxlen=max_token)
  if np.around(model.predict(X)[0])==1:
    print("=============> SPAM\n")
    return 0
  else:
    print("=============> HAM\n")
    return 1
  

def predict(text,model,tfidf_vector):
  print(text)
  text=standardize_data(text)
  token=remove_stopwords(text)
  w_lenmatizer=word_lenmatizer(token)
  w_lenmatizer = [" ".join(x) for x in w_lenmatizer]
  X_Tfidf = tfidf_vector.transform(w_lenmatizer)
  if model.predict(X_Tfidf)[0]==1:
    print("=============> SPAM\n")
    return 0
  else:
    print("=============> HAM\n")
    return 1

def multi_predict(path,model,tokenizer,choice):
  df=load_data(path)
  hl_tokens = []
  for hl in df['CONTENT']:
      hl=standardize_data(hl)
      hl=remove_stopwords(hl)
      hl_tokens.append(hl)
  # Init Lemmatizer
  lemmatizer = WordNetLemmatizer()
  hl_lemmatized = []
  for tokens in hl_tokens:
      lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
      hl_lemmatized.append(lemm)

  if choice==0:
      sequences = tokenizer.texts_to_sequences(hl_lemmatized)
      X = pad_sequences(sequences, maxlen=110)
      Y = df['CLASS'].values
      Y = np.vstack(Y)
      pred=np.around(model.predict(X))
      results = []
      for text in df['CONTENT']:
            # 0 is SPAM, 1 NO SPAM
            result = []
            res = LSTM_predict(text, model, tokenizer)
            result.append(text)
            result.append(res)
            results.append(result)
      return classification_report(Y,pred), results
  else:
      X = [" ".join(x) for x in hl_lemmatized]
      Test_X_Tfidf = tokenizer.transform(X)
      Y = df['CLASS'].values
      pred=model.predict(Test_X_Tfidf)
      results = []
      
      for text in df['CONTENT']:
            # 0 is SPAM, 1 NO SPAM
            result = []
            res = predict(text, model, tokenizer)
            result.append(text)
            result.append(res)
            results.append(result)
      return classification_report(Y,pred), results

LSTM_model = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/LSTM_model.pkl')
LSTM_TOKEN = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/tokenizer_LSTM.pkl')
NB_model = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/NB_model.pkl')
SVM_model = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/SVM_model.pkl')
tfidf = joblib.load('/Users/lethachlam/Developer/Datamining-Project/model/tfidf.pkl')

@app.route('/lstm',methods = ['POST'])
def pridictLSTM():

      tb._SYMBOLIC_SCOPE.value = True

      text = str(request.get_json('DATA')['DATA'])

      x = LSTM_predict(text, LSTM_model, LSTM_TOKEN)

      one, two, three = pre_processing(text)

      convert_vector = vectorize_lstm(text, LSTM_TOKEN)[0]

      string_vector = np.array_str(convert_vector)

      data = {
        'Result': x,
        'standardize': one,
        'tokens': two,
        'lenmatizer': str(three),
        'vector': string_vector
      }

      return jsonify(data)

@app.route('/svm',methods = ['POST'])
def pridictModelSVM():
            tb._SYMBOLIC_SCOPE.value = True

            text = str(request.get_json('DATA')['DATA'])

            x = predict(text, SVM_model,tfidf)

            one, two, three = pre_processing(text)
            
            convert_vector = vectorize_clasifer(three,tfidf)

            string_vector = str(convert_vector)

            if x is None:
                x = "NULL"
            
            data = {
              'Result': x,
              'standardize': one,
              'tokens': two,
              'lenmatizer': str(three),
              'vector': string_vector
            }

            return jsonify(data)

@app.route('/np',methods = ['POST'])
def pridictModelNP():
            tb._SYMBOLIC_SCOPE.value = True

            text = str(request.get_json('DATA')['DATA'])

            x = predict(text, NB_model,tfidf)

            one, two, three = pre_processing(text)

            convert_vector = vectorize_clasifer(three,tfidf)

            string_vector = str(convert_vector)

            if x is None:
                x = "NULL"
            data = {
              'Result': x,
              'standardize': one,
              'tokens': two,
              'lenmatizer': str(three),
              'vector': string_vector
            }

            return jsonify(data)


@app.route('/',methods = ['GET'])
def home(name = None):
    return render_template('upload_file.html', name=name)

@app.route('/multipredict',methods = ['POST'])
def multipredictAll():
    tb._SYMBOLIC_SCOPE.value = True
    method = request.get_json()['method']
    id = request.get_json()['id']
    print(method)
    print(id)
    model = None
    token = None
    path = ""
    choice = 1
    if method == "LSTM":
          model = LSTM_model
          token = LSTM_TOKEN
          choice = 0
    elif method == "NB":
          model = NB_model
          token = tfidf
    elif method == "SVM":
          model = SVM_model
          token = tfidf

    if id == 0:
          path = "/Users/lethachlam/Developer/Datamining-Project/data/list1"
    else:
          path = "/Users/lethachlam/Developer/Datamining-Project/data/list2"
    results = []
    x, results = multi_predict(path,model,token,choice)

    data = {
      "Multi": x,
      "results": results
    }
    return jsonify(data)

@app.route('/multipredict',methods = ['GET'])
def multipridict():
    x = multi_predict('/Users/lethachlam/Developer/Datamining-Project/data',LSTM_model,LSTM_TOKEN,0)
    data = {
      "Multi": x
    }
    return jsonify(data)
if __name__ == '__main__':
    # host iphone = 172.20.10.2
    # host='192.168.43.239',
    app.run(host = '127.0.0.1', port = '1998',debug=True, threaded=False)
