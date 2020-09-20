#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[2]:


with open("train_qa.txt", "rb") as fp:   # Unpickling
    train_data =  pickle.load(fp)


# In[3]:


with open("test_qa.txt", "rb") as fp:   # Unpickling
    test_data =  pickle.load(fp)


# In[4]:


story = " ".join(train_data[0][0])


# In[5]:


qn = " ".join(train_data[0][1])


# In[6]:


ans = train_data[0][2]


# In[7]:


print(f"Story :{story} Question:{qn} Answer is {ans}")


# In[8]:


vocab_= set()


# In[9]:


data = train_data + test_data


# In[10]:


for story,qn,ans in data:
    vocab_ = vocab_.union(story)
    vocab_ = vocab_.union(qn)


# In[11]:


vocab_.add("no")
vocab_.add("yes")
voc = "|".join(vocab_)


# In[12]:


print("Hi I am Bot.I know those words given below.you can make story with those words and ask me the question.I will try to answer it.")
print(voc,end="")


# In[14]:


vocab_size = len(vocab_) + 1
print(vocab_size)


# In[15]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[17]:


tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab_)


# In[19]:


train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in data:
    train_story_text.append(story)
    train_question_text.append(question)


# In[21]:


len(train_story_text)


# In[22]:


train_story_seq = tokenizer.texts_to_sequences(train_story_text)


# In[24]:


len(train_story_seq)


# In[36]:


max_story_len = max([len(data[0]) for data in data])
max_question_len = max([len(data[1]) for data in data])
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,max_question_len=max_question_len):
    x = []
    y = []
    z = []
    for story, query, answer in data:
        sto = [word_index[word.lower()] for word in story]
        
        qn = [word_index[word.lower()] for word in query]
        
        ans = np.zeros(len(word_index) + 1)
        
        ans[word_index[answer]] = 1
        
        x.append(sto)
        
        y.append(qn)
        
        z.append(ans)
        
    return (pad_sequences(x, maxlen=max_story_len),pad_sequences(y, maxlen=max_question_len), np.array(z))


# In[38]:


inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)


# In[53]:





# In[40]:


from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM


# In[41]:


input_sequence = Input((max_story_len,))
question = Input((max_question_len,))


# In[42]:


input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))


# In[43]:


input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))


# In[45]:


question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_question_len))
question_encoder.add(Dropout(0.3))


# In[46]:


input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


# In[47]:


match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)


# In[48]:


response = add([match, input_encoded_c])
response = Permute((2, 1))(response)
answer = concatenate([response, question_encoded])


# In[50]:


answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)


# In[51]:


print("------------------------------")
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[54]:


history = model.fit([inputs_train, queries_train], answers_train,batch_size=32,epochs=120,validation_data=([inputs_test, queries_test], answers_test))
filename = 'storyqn_anschatbot.h5'
model.save(filename)


# In[ ]:





# In[ ]:




