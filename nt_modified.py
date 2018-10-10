from __future__ import print_function
""" First change the following directory link to where all input files do exist """
import os
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from keras.models import load_model
import json
from keras.layers import Input, Dense, Dropout
from keras.models import Model






# File reading
class sequence_word_prediction:
    def __init__(self):
        print("intitialized sequence word prediction")
#        self.content = content
#        self.x_trigm = x_trigm
#        self.y_trigm = y_trigm
#        self.X = X
#        self.Y = Y
#        self.Xtrain = Xtrain
#        self.Ytrain = Ytrain
    def Preprocessing(self):
        with open('./iSOLAR.csv', 'r') as content_file:
            content = content_file.read()
        content2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in content]).split())
    # Tokenize content2
        tokens = nltk.word_tokenize(content2)
        tokens = [word.lower() for word in tokens if len(word)>=3]
    # Select value of N for N grams among which N-1 are used to predict last N word
        N = 2
        quads = list(nltk.ngrams(tokens, N))
        #print(quads[:5])
        newl_app = []
        for ln in quads:
            newl = " ".join(ln)        
            newl_app.append(newl)
           # print(newl_app[:5])
    
        x_trigm = []
        y_trigm = []
    
        for l in newl_app:
            x_str = " ".join(l.split()[0:N-1])  
            y_str = l.split()[N-1]   
            x_trigm.append(x_str)
            y_trigm.append(y_str)
        return x_trigm, y_trigm
    def vectorizing(self,x_trigm, y_trigm):
        vectorizer = CountVectorizer()
        x_trigm_check = vectorizer.fit_transform(x_trigm).todense()
        y_trigm_check = vectorizer.fit_transform(y_trigm).todense()
       # print(x_trigm_check.shape)
       # print(y_trigm_check.shape)
    # Dictionaries from word to integer and integer to word
        dictnry = vectorizer.vocabulary_
        X = np.array(x_trigm_check)
        Y = np.array(y_trigm_check)
        return X,Y,dictnry
    def splitting_data(self,X,Y,x_trigm):
        Xtrain, Xtest, Ytrain, Ytest, xtrain_tg, xtest_tg = train_test_split(X, Y, x_trigm, test_size=0.2, random_state=42)
        #print("X Train shape",Xtrain.shape, "Y Train shape", Ytrain.shape)
        #print("X Test shape",Xtest.shape, "Y Test shape", Ytest.shape)
        return Xtrain, Xtest, Ytrain, Ytest, xtrain_tg, xtest_tg
    def Model_building(self,Xtrain,Ytrain):
        np.random.seed(42)
        input_layer = Input(shape = (Xtrain.shape[1],),name="input")
        print(input_layer.shape)
        first_layer = Dense(1000,activation='relu',name = "first")(input_layer)
        first_dropout = Dropout(0.5,name="firstdout")(first_layer)
        second_layer = Dense(800,activation='relu',name="second")(first_dropout)
        third_layer = Dense(1000,activation='relu',name="third")(second_layer)
        third_dropout = Dropout(0.5,name="thirdout")(third_layer)
        fourth_layer = Dense(Ytrain.shape[1],activation='softmax',name = "fourth")(third_dropout)
        print(fourth_layer.shape)
        history = Model(input_layer,fourth_layer)
        history.compile(optimizer = "adam",loss="categorical_crossentropy",metrics=["accuracy"])
        print (history.summary())
        BATCH_SIZE = 128
        NUM_EPOCHS = 100
        #history.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_split = 0.2)
        
        return history

# Model Prediction
    def Model_prediction(self,history, Xtest):
        Y_pred = history.predict(Xtest)
        print(Y_pred.shape)
        print("Prior bigram words", "|Actual", "|Predicted","\n")
        return Y_pred


        return Y_pred
    def Post_processing(self, txt_input,dictnry,Y_pred, history):
        txt_input_content = " ".join("".join([" " if ch in string.punctuation else ch for ch in txt_input]).split())
        txt_input_tokens = nltk.word_tokenize(txt_input_content)
        txt_input_tokens = [word.lower() for word in txt_input_tokens if len(word)>=3]
        for i in dictnry.keys():
            dictnry[i] = int(dictnry[i])
        arr_shape = Y_pred.shape[1]
        test_array = np.zeros(arr_shape, dtype=int)
        history.save("./model_saved.h5")
        for i in txt_input_tokens:
            if i in dictnry.keys():
                test_array[dictnry[i]] = 1
        my_model = load_model("./model_saved.h5")
        my_pred = my_model.predict(test_array.reshape((1,arr_shape)))
        rev_dictnry = {v:k for k,v in dictnry.items()}
        output = rev_dictnry[np.argmax(my_pred)]
        return output
    def funct(self, txt_input):
        x_trigm,y_trigm = self.Preprocessing()
        X,Y,dictnry =  self.vectorizing(x_trigm, y_trigm)
        Xtrain, Xtest, Ytrain, Ytest, xtrain_tg, xtest_tg = self.splitting_data(X,Y,x_trigm)
        history = self.Model_building(Xtrain,Ytrain)
        Y_pred = self.Model_prediction(history, Xtest)
        output = self.Post_processing(txt_input,dictnry,Y_pred, history)
        return output
       
#txt_input = "immediately"
#model_check = sequence_word_prediction()
#print(model_check.funct(txt_input))
#print(model_check.validation_test_data())  

