#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pickle
import numpy as np


# In[81]:


with open("train_qa.txt", "rb") as fp:   # Unpickling
    train_data =  pickle.load(fp)


# In[82]:


with open("test_qa.txt", "rb") as fp:   # Unpickling
    test_data =  pickle.load(fp)


# In[83]:


vocab_= set()


# In[84]:


data = train_data + test_data


# In[85]:


for story,qn,ans in data:
    vocab_ = vocab_.union(story)
    vocab_ = vocab_.union(qn)


# In[86]:


vocab_.add("no")
vocab_.add("yes")
voc = ", ".join(vocab_)


# In[87]:


print("Hi I am Bot.I know those words given below.you can make story with those words and ask me the question.I will try to answer it.")
print()
print()
print(voc,end="")


# In[88]:


vocab_size = len(vocab_) + 1


# In[89]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[90]:


tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab_)


# In[91]:


train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in data:
    train_story_text.append(story)
    train_question_text.append(question)


# In[92]:


train_story_seq = tokenizer.texts_to_sequences(train_story_text)


# In[93]:


max_story_len = max([len(data[0]) for data in data])
max_question_len = max([len(data[1]) for data in data])
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,max_question_len=max_question_len):
    x = []
    y = []
    
    for story, query in data:
        sto = [word_index[word.lower()] for word in story]
        
        qn = [word_index[word.lower()] for word in query]
        
        x.append(sto)
        
        y.append(qn)
        
        
    return (pad_sequences(x, maxlen=max_story_len),pad_sequences(y, maxlen=max_question_len))


# In[102]:


from keras.models import load_model
filename = 'chatbot_120_epochs.h5'
model = load_model(filename)


# In[109]:


my_story = input()
my_story.split()
my_question = input()
my_question.split()
mydata = [(my_story.split(),my_question.split())]
my_story,my_ques = vectorize_stories(mydata)


# In[110]:


pred_results = model.predict(([ my_story, my_ques]))


# In[111]:


val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key
        
print("I guess the answer is",k)


# In[ ]:




