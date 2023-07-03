# -*- coding: utf-8 -*-
# import flask dependencies
try:
    import urllib
    import json
    import os
    from flask import (Flask,request,jsonify)
    import numpy as np
    import re
    import nltk
    # nltk.download('stopwords')
    import pickle
    from nltk.corpus import stopwords
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import gc
    #import boto3
    #from tf.keras.models import Model
    #from tf.keras.models import load_model

except Exception as e:

    print("Some modules are missing {}".format(e))


#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('omw-1.4')





# initialize the flask app
app = Flask(__name__)


# create a route for webhook

@app.route('/webhook', methods=['GET','POST'])
def webhook():
    req = request.get_json(silent = True, force = True)
    query_result = req.get('text')
    res = get_data(query_result)
    return res


#################################
def get_wordnet_pos(word):
    pack = nltk.pos_tag([word])
    tag = pack[0][1]
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def replace(new_sentence):
    replacement_patterns = [
      (r'won\'t', 'will not'),
      (r'can\'t', 'cannot'),
      (r'i\'m', 'i am'),
      (r'ain\'t', 'is not'),
      (r'(\w+)\'ll', '\g<1> will'),
      (r'(\w+)n\'t', '\g<1> not'),
      (r'(\w+)\'ve', '\g<1> have'),
      (r'(\w+)\'s', '\g<1> is'),
      (r'(\w+)\'re', '\g<1> are'),
      (r'(\w+)\'d', '\g<1> would')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in replacement_patterns]
    
    for (pattern, repl) in patterns:
        (new_sentence, count) = re.subn(pattern, repl, new_sentence)
    return new_sentence

def remove_pun(new_sentence):
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    new_sentence = re.sub(r"[%s]+" %punc, "",new_sentence)
    new_sentence = new_sentence.replace('\n', '').replace('\r', '')
    return new_sentence

def digit_remove(new_sentence):
    new_sentence =  " ".join([word for word in new_sentence.split() if not word.isdigit()])
    return new_sentence

def stopwords_remove(sentence):
    #stoplist = stopwords.words('english')
    with open('stopwords.txt') as file:
        stoplist = [stopword.replace('\n', '').lower() for stopword in file.readlines()]
    new_sentence = [word for word in sentence if word not in stoplist]
    return new_sentence

def lemmatize(sentence):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    new_sentence = [lemmatizer.lemmatize(word, get_wordnet_pos(word) or wordnet.NOUN) for word in sentence]
    return new_sentence

def transfomrer_text(new_sentence_lemmatize):
    MAX_SEQUENCE_LENGTH = 1000
    with open('text_tokenizer', 'rb') as training_model:
        tokenizer = pickle.load(training_model)
    test_sequences = tokenizer.texts_to_sequences([new_sentence_lemmatize])
    del tokenizer 
    gc.collect()
    x_predict = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH) # 1000
    return x_predict

def preprocess_text(textdata):
    new_sentence_lower = textdata.lower()
    new_sentence_replace = replace(new_sentence_lower)
    new_sentence_nopun = remove_pun(new_sentence_replace)
    new_sentence_nodigit = digit_remove(new_sentence_nopun)
    new_sentence_tokenize = nltk.word_tokenize(new_sentence_nodigit)
    new_sentence_stopwords = stopwords_remove(new_sentence_tokenize)
    new_sentence_lemmatize = lemmatize(new_sentence_stopwords)
    x_predict = transfomrer_text(new_sentence_lemmatize)
    return x_predict



################# original code 16 types##########################

# def classifier_predict1(x_predict):
#     TypeDict=['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP','INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP']
#     model = load_model('classifier.h5')
#     y_prob = model.predict(x_predict)
#     y_classes = y_prob.argmax(axis=-1)
#     return TypeDict[y_classes[0]]

# def get_data(textdata):
#     x_predict = preprocess_text(textdata)
#     classification_res = classifier_predict(x_predict)

#########################################################

def classifier_predict1(x_predict):
    model1 = load_model('25_IE_model_rnn .hdf5')
    y_prob1 = model1.predict(x_predict)
    y_classes1 = y_prob1.argmax(axis=-1)
    ei = ['E', 'I']
    del model1
    gc.collect()
    return ei[y_classes1[0]]

def classifier_predict2(x_predict):
    model2 = load_model('25_SN_model_rnn.hdf5')
    y_prob2 = model2.predict(x_predict)
    y_classes2 = y_prob2.argmax(axis=-1)
    sn = ['N', 'S']
    del model2
    gc.collect()
    return sn[y_classes2[0]]

def classifier_predict3(x_predict):
    model3 = load_model('25_FT_model_rnn.hdf5')
    y_prob3 = model3.predict(x_predict)
    y_classes3 = y_prob3.argmax(axis=-1)
    ft = ['F', 'T']
    del model3
    gc.collect()
    return ft[y_classes3[0]]

def classifier_predict4(x_predict):
    model4 = load_model('25_PJ_model_rnn.hdf5')
    y_prob4 = model4.predict(x_predict)
    y_classes4 = y_prob4.argmax(axis=-1)
    pj = ['J', 'P']
    del model4
    gc.collect()
    return pj[y_classes4[0]]

def get_data(textdata):
    x_predict = preprocess_text(textdata)
    classification_res1 = classifier_predict1(x_predict)
    classification_res2 = classifier_predict2(x_predict)
    classification_res3 = classifier_predict3(x_predict)
    classification_res4 = classifier_predict4(x_predict)
    classification_res = classification_res1+classification_res2+classification_res3+classification_res4
    print(classification_res)


#########################################################
    response_dict_ESTJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ESTJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/a78aaca9-f77a-433d-a641-0bc317d62392"
        }
    
    response_dict_ESTP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ESTP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/9bce7032-086b-4ff7-a50b-7da71b251a81"
        }
    
    response_dict_ESFJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ESFJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/0e9990bc-6852-48ae-a85b-83ba215a61e8"
        }
    
    response_dict_ESFP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ESFP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/77e08929-99cc-4b7b-af95-dc9506a52c54"
        }
    
    response_dict_ENTJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ENTJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/699d85bc-3ba3-4046-aef2-78c523714129"
        }
    
    response_dict_ENTP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ENTP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/96aaed72-031e-4161-9894-f528de32f089"
        }
    
    response_dict_ENFJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ENFJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/acdaa6d4-bde7-4619-85ef-9db6f2083e8c"
        }
    
    response_dict_ENFP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ENFP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/e258ff4a-b850-4875-89a0-7064af53a9ee"
        }
    
    response_dict_ISTJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ISTJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/bb940950-ea35-4ed2-b696-8adca327fc9c"
        }
    
    response_dict_ISTP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ISTP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/71b9af5a-67b1-4d90-b54c-cb90e8ac0c6e"
        }  
    
    response_dict_ISFJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ISFJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/cd0eb7fc-1907-4de3-9151-9ec48370b02b"
        }
    
    response_dict_ISFP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is ISFP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/36b33022-3482-43c2-9be6-98b74a9b644c"
        }
    
    response_dict_INTJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is INTJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/6198a30e-46f6-423d-a77d-2c82ac2a0c9d"
        }
    
    response_dict_INTP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is INTP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/250794e9-da41-49ce-8b23-364c861a7287"
        }
    
    response_dict_INFJ = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is INFJ!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/e9dc9ab5-4ecc-4fc8-9dcb-52f0145d3f2b"
        }
    
    response_dict_INFP = {
        'fulfillmentResponse':{
            'messages':[{
                'text':{
                    'text': ["Hey! Your personality type is INFP!"],
                    'allowPlaybackInterruption': False
                    }
                }],
            'mergeBehavior':"MERGE_BEHAVIOR_UNSPECIFIED"
            },
        'pageInfo':request.json['pageInfo'],
        'sessionInfo':request.json['sessionInfo'],
        'targetPage': "projects/chatbot1-ydcj/locations/us-central1/agents/8f54bb45-cd9e-4ca1-a214-75604809c944/flows/00000000-0000-0000-0000-000000000000/pages/ad9c08c0-472f-4a27-83d0-6efbcf405740"
        }
    
    
    if classification_res == 'ESTJ':
        return jsonify(response_dict_ESTJ)
    elif classification_res == 'ESTP':
        return jsonify(response_dict_ESTP)
    elif classification_res == 'ESFJ':
        return jsonify(response_dict_ESFJ)
    elif classification_res == 'ESFP':
        return jsonify(response_dict_ESFP)
    elif classification_res == 'ENTJ':
        return jsonify(response_dict_ENTJ)
    elif classification_res == 'ENTP':
        return jsonify(response_dict_ENTP)
    elif classification_res == 'ENFJ':
        return jsonify(response_dict_ENFJ)
    elif classification_res == 'ENFP':
        return jsonify(response_dict_ENFP)
    elif classification_res == 'ISTJ':
        return jsonify(response_dict_ISTJ)
    elif classification_res == 'ISTP':
        return jsonify(response_dict_ISTP)
    elif classification_res == 'ISFJ':
        return jsonify(response_dict_ISFJ)
    elif classification_res == 'ISFP':
        return jsonify(response_dict_ISFP)
    elif classification_res == 'INTJ':
        return jsonify(response_dict_INTJ)
    elif classification_res == 'INTP':
        return jsonify(response_dict_INTP)
    elif classification_res == 'INFJ':
        return jsonify(response_dict_INFJ)
    else:
        return jsonify(response_dict_INFP)
    

# run the app
if __name__ == '__main__':
    app.run()
    
