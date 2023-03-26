#!/usr/bin/env python
# coding: utf-8

# ## Yoruba Language Detection
# 
# ##### Final Year Project

# ### Importing Basic Libraries

# In[4]:


import string
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_csv('language_detection.csv')
df.head()


# ### Cleaning up the dataset, using string library 

# In[6]:


string.punctuation


# ### A function that cleans the dataset

# In[7]:


def remove_pun(text):
    for pun in string.punctuation:
        text = text.replace(pun,"")
    text = text.lower()
    return(text)
        


# ### Trying out the function

# In[8]:


remove_pun('"Nature can refer to the phenomena of the: 44##@! physical@."')


# In[9]:


remove_pun('"lílo àkàbà — ǹjẹ́ o máa ń ṣe àyẹ̀wò wọ̀nyí tó lè dáàbò bò ẹ́,? re"') # => Working wellwith yoruba alphabets


# ### Applying the Function on our Dataset
# 
# ###### This removes every punctuation in the dataset and converts to lowercase

# In[10]:


df['Text'] = df['Text'].apply(remove_pun)


# In[11]:


df.head()


# In[12]:


df.shape


# ### Dividing datasets to train and test

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X = df.iloc[:,0] # => Assigning the Texts to X
y =df.iloc[:,1] # => Assigning the Language Column
# X
# Y


# ### Assigning test and train data

# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = .2)
# X_train,X_test,y_train,y_test


# ### Converting values to computer understandable version = Encoding
# 
# ###### Vectorizing the dataset

# In[16]:


from sklearn import feature_extraction


# In[17]:


vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1,2),analyzer='char') # Unigrams and bigrams


# In[18]:


from sklearn import pipeline
from sklearn import linear_model


# ### pipeline: creating a complete flow of functions (converting to vector and training) multpile steps

# In[19]:


model_pipe = pipeline.Pipeline([('vec',vec),('clf', linear_model.LogisticRegression())])
# model_pipe


# In[20]:


model_pipe.fit(X_train,y_train)


# In[21]:


model_pipe.classes_


# In[22]:


predict_val = model_pipe.predict(X_test)
# predict_val


# 
# ### Calculating the Accuracy 

# In[23]:


from sklearn import metrics


# In[24]:


metrics.accuracy_score(y_test,predict_val) #99% Accuracy, *100


# In[25]:


metrics.confusion_matrix(y_test,predict_val)


# In[26]:


model_pipe.predict(['My name is osas']) 


# In[29]:


model_pipe.predict(['is my friend'])


# ### Saving as a pickle file
# 
# ##### to be used on the web

# In[30]:


import pickle


# In[32]:


new_file = open('model.pckl', 'wb')
new_file = open('model.pkl', 'wb')
pickle.dump(model_pipe,new_file)
new_file.close()


# In[ ]:





# ## Thank You!!!
