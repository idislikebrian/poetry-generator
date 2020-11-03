from bs4 import BeautifulSoup
import requests
import os
import time

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np

## data mining ##
element = "https://poets.org/poems/pablo-neruda" ## insert poet's page from poets.org
poet = requests.get(element)
poetData = BeautifulSoup(poet.content, 'html.parser')
rowData = poetData.find("tbody", attrs={"role":"rowgroup"})
poems = rowData.find_all("td", attrs={"data-label":"Title"})
poemList = []
for tag in rowData.find_all('a', href=True):
    poemList.append(str(tag['href']))
counter = 0
for x in poemList:
    if counter %2 == 0:
        time.sleep(3)
        poemURL = "https://poets.org{}".format(x)
        search = requests.get(poemURL)
        poetry = BeautifulSoup(search.content, 'html.parser')
        poemTitle = poetry.find("h1", attrs={"class":"card-title"})
        lyrics = poetry.find("div", attrs={"class":"poem__body px-md-4 font-serif"})
        ## append to text file ##
        f = open("src\poems\data\pablo-neruda.txt", "a")
        f.write(str(poemTitle.text) + "\n \n" + str(lyrics.text) + "\n" + "---" + "\n" )
    counter += 1

## learning ##
data = open('src\poems\data\pablo-neruda.txt').read()
data[0:300]

tokenizer = Tokenizer()
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print('Total number of words in corpus:',total_words)

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)
# get max sequence length
max_sequence_len = max([len(x) for x in input_sequences])
#pad the sequence
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)

# Defining the model.
model = Sequential()
model.add(Embedding(total_words,100,input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150,return_sequences=True)))
model.add(Dropout(0.18))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(total_words/2,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
print(model.summary())

# Training the model.
history = model.fit(predictors, label, epochs=100, verbose=1)

# Testing the model.
seed_text = "Cold water"
next_words = 60
for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-    1, padding='pre')
  predicted = model.predict_classes(token_list, verbose=0)
  output_word = ""
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break
  seed_text += " " + output_word
print(seed_text)