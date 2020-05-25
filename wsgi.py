from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
import string
import nltk
from string import digits, punctuation
from flask import Flask
from flask_restful import Resource, Api
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import argparse
import os
from flask import request, jsonify
import keras.backend.tensorflow_backend as tb
import socket

app = Flask(__name__)
ip = socket.gethostbyname(socket.gethostname())

# Example comparison
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
    return False
  else:
    print(text)
    print("=============> HAM\n")
    return True
  

def predict(text,model,tfidf_vector):
  w_lenmatizer=word_lenmatizer(text)
  w_lenmatizer = [" ".join(x) for x in w_lenmatizer]
  X_Tfidf = tfidf_vector.transform(w_lenmatizer)
  if model.predict(X_Tfidf)[0]==1:
    print(text)
    print("=============> SPAM\n")
  else:
    print(text)
    print("=============> HAM\n")


def testItem():
      text = 'Please subcribe my channel'
      LSTM_model = pickle.loads(open('/Users/lethachlam/Developer/Flask_Youtube/model/LSTM_model.pkl', "rb").read())
      LSTM_TOKEN = pickle.loads(open('/Users/lethachlam/Developer/Flask_Youtube/model/tokenizer_LSTM.pkl', "rb").read())
      X = LSTM_predict(text, LSTM_model,LSTM_TOKEN)
      if X == True:
            print('1')
            return {
              'DATA':['1']
            }
      else: 
        print(0)
        return {
        'DATA':['0']
      }
# text = 'Please subcribe my channel'

LSTM_model = pickle.loads(open('/Users/lethachlam/Developer/Flask_Youtube/model/LSTM_model.pkl', "rb").read())
LSTM_TOKEN = pickle.loads(open('/Users/lethachlam/Developer/Flask_Youtube/model/tokenizer_LSTM.pkl', "rb").read())
class PridictLSTM(Resource):
    async def get(self, text):
            X = await LSTM_predict(text, LSTM_model,LSTM_TOKEN)
            if X == True:
                  print('1')
                  return {
                    'DATA':['1']
                  }
            else: 
              print(0)
              return {
              'DATA':['0']
            }     
class Quotes(Resource):
    def get(self):
        return {
            'William Shakespeare': {
                'quote': ['Love all,trust a few,do wrong to none',
                'Some are born great, some achieve greatness, and some greatness thrust upon them.']
        },
        'Linus': {
            'quote': ['Talk is cheap. Show me the code.']
            }
        }
class TodoSimple(Resource):
    def get(self, text):
        return {
          'DATA': text
        }
    def put(self, text):
        return {"data": text}
# api.add_resource(Quotes, '/predict')
# api.add_resource(PridictLSTM,'/<string:text>')
# api.add_resource(TodoSimple,'/api/<string:text>')
@app.route('/test',methods = ['POST'])
def pridictTest():
      tb._SYMBOLIC_SCOPE.value = True
      text = str(request.get_json('DATA'))
      x = LSTM_predict(text, LSTM_model,LSTM_TOKEN)
      data = {
        'Result': x
      }
      return jsonify(data)

if __name__ == '__main__':
    app.run()