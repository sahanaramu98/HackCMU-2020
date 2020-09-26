import numpy as np
#from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request, redirect, url_for
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pickle
import pandas as pd
import sklearn


import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

glove2word2vec(glove_input_file="/home/lenovo/Documents/Hackathon/glove.6B/glove.6B.100d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

vocab = glove_model.vocab.keys()

# def create_model():
#     model = Sequential()

#     # model.add(layers.Conv2D(filters=6, kernel_size=(3, 1), strides=(1,1), activation='relu', input_shape=input_shape))
#     # model.add(layers.AveragePooling2D())

#     # model.add(layers.Conv2D(filters=16, kernel_size=(3, 1), strides=(1,1), activation='relu'))
#     # model.add(layers.AveragePooling2D())

#     model.add(layers.Flatten())
#     model.add(layers.Dense(units=900, activation='relu'))
#     model.add(layers.Dense(units=256, activation='relu'))
#     model.add(layers.Dense(units=32, activation='relu'))
#     model.add(layers.Dense(units=2, activation = 'softmax'))
    
#     return model

app = Flask(__name__)
# model = create_model()
# model.load_weights("twitter_sentiment_100d.h5")
model = tf.keras.models.load_model("twitter_sentiment_100d.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

import requests

import numpy as np


def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )


operations = dict([
    ('mean', (lambda word_embeddings: [np.mean(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('max', (lambda word_embeddings: [np.max(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('min', (lambda word_embeddings: [np.min(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('p_mean_2', (lambda word_embeddings: [gen_mean(word_embeddings, p=2.0).real], lambda embeddings_size: embeddings_size)),
    ('p_mean_3', (lambda word_embeddings: [gen_mean(word_embeddings, p=3.0).real], lambda embeddings_size: embeddings_size)),
])


def get_sentence_embedding(sentence, embeddings, chosen_operations):
    word_embeddings = []
    for tok in sentence:
        if tok not in vocab:
            continue
        vec = embeddings[tok]
        if vec is not None:
            word_embeddings.append(vec)

    if not word_embeddings:
        sentence_embedding = np.zeros(300)
    else:
        concat_embs = []
        for o in chosen_operations:
            concat_embs += operations[o][0](word_embeddings)
        sentence_embedding = np.concatenate(
            concat_embs,
            axis=0
        )

    return sentence_embedding

def get_embedding(sent):
    return get_sentence_embedding(sent.split(), glove_model, ['min', 'mean', 'max'])

def NewsFromBBC(query): 

    # BBC news api 
    main_url = "https://newsapi.org/v2/everything?"
    parameters = {
    'q': query, # query phrase
    'pageSize': 5,  # maximum is 100
    'apiKey': '38c5b69ec17c41b5aa61f1d6467f18ef' # your own API key
    }
    # fetching data in json format 
    open_bbc_page = requests.get(main_url, params=parameters).json() 

    # getting all articles in a string article 
    article = open_bbc_page["articles"] 

    # empty list which will
    # contain all trending news 
    results = [] 
    links = []

    for ar in article: 
        results.append(ar["title"]) 
        links.append(ar["url"])

#     for i in range(len(results)): 

#         # printing all trending news 
#         print(i + 1, results[i])

    return results, links

# Driver Code 
# if name == 'main': 

    # function call 


def pre_embedd(results):
    df = pd.DataFrame(results)
    df[0] = df[0].str.split(' ').apply(lambda x: ' '.join([k for k in x if not (('http://' in k) or ('.com'  in k) or ('@'  in k) or ('#'  in k))]))
    df[0] = df[0].str.replace(r'[^a-zA-Z\s]','').str.lower()
    test_data = df[0].apply(lambda x: get_embedding(x))
    X_test = []
    for index in range(len(test_data)):
        X_test.append(test_data[index])
    
    X_test = np.array(X_test)
    return X_test

def predict1(X_test):
    classes = model.predict(X_test)
    return np.argmax(classes, axis=1)

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict/<query>',methods=['GET'])
def predict(query):
    results, links = NewsFromBBC(query) 
    X_test = pre_embedd(results)
    predicted = predict1(X_test)
    output = ["positive" if i == 1 else "negative" for i in predicted ]
    
    #outstr = "" 
    
    #print(output)
    l = []
    h = []
    for index, news in enumerate(results):
        #outstr = outstr + '<a href="{}">'.format(links[index]) + news + "</a>" +  " &nbsp "  + "<br/>"
        l.append(links[index])
        h.append(news)
    #return outstr
    return render_template("output.html", link1=l[0], link2=l[1], link3=l[2], link4=l[3], link5=l[4], news1=h[0], news2=h[1], news3=h[2], news4=h[3], news5=h[4], output1 = output[0], output2 = output[1], output3 = output[2], output4 = output[3], output5 = output[4])

    #, link=links[index],output=output[index],news1=news
    #return np.array_str(predicted)
    #str(output[index])

    #return render_template('index.html', prediction_text='Sales should be $ {}'.format(outstr))

#@app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))

# get the data for the requested query
@app.route('/success/<name>')
def success(name):
    #return "<xmp>" + str(predict(name)) + " </xmp> "
    return str(predict(name))


if __name__ == "__main__":
    app.run(debug=True)