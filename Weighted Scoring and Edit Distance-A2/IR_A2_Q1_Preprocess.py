#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import email
import pickle
import math

import nltk
import re, unicodedata
import contractions
import inflect

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer


# In[2]:


def dataLoad(filename):
    file1=open(filename,"r+",encoding='ansi')
    file_payload=file1.read()
    return file_payload


# In[3]:


def accentedCharacters(file_payload):
    return unicodedata.normalize('NFKD', file_payload).encode('ascii', 'ignore').decode('utf-8', 'ignore')


# In[4]:


def specialCharactersRemoval(file_payload):
    #Assumption that decimal number will not be retained and converted to regular number without .
    pattern1=r'[^a-zA-z0-9\s]'
    result1=re.sub(pattern1, '', file_payload)
    
    pattern2=r'[\\\_\]\(\)\^]'#Regex metacharacter match and then removal which usually dont removes above
    result2=re.sub(pattern2,'', result1)
    
    return result2


# In[5]:


def expandContractions(file_payload):
    return contractions.fix(file_payload)


# In[6]:


def toLowerCase(file_payload):
    return file_payload.lower()


# In[7]:


def tokenizer(file_payload):
    return nltk.word_tokenize(file_payload)


# In[14]:


def numberToWords(file_tokenized_list):
    iengine=inflect.engine()
    for i in range(len(file_tokenized_list)):
        if file_tokenized_list[i].isdigit() and int(file_tokenized_list[i])<99999:
            file_tokenized_list[i]=iengine.number_to_words(file_tokenized_list[i])
        else:
            file_tokenized_list[i]=file_tokenized_list[i]
    return file_tokenized_list


# In[9]:


def stopWordRemoval(file_tokenized_list):
    file_tokenized_list_new=[]
    for i in range(len(file_tokenized_list)):
        if file_tokenized_list[i] not in stopwords.words('english'):
            file_tokenized_list_new.append(file_tokenized_list[i])
    return file_tokenized_list_new


# In[10]:


def singleCharRemoval(file_tokenized_list):
    return [i for i in file_tokenized_list if len(i) > 1]  


# In[11]:


def lemmatization(file_tokenized_list):
    file_tokenized_list_new=[]
    lemmatizer=WordNetLemmatizer()
    for token in file_tokenized_list:
        if lemmatizer.lemmatize(token)!=token:
            file_tokenized_list_new.append(lemmatizer.lemmatize(token))
        elif lemmatizer.lemmatize(token,'v')!=token:
            file_tokenized_list_new.append(lemmatizer.lemmatize(token,'v'))
        else:
            file_tokenized_list_new.append(token)
    return file_tokenized_list_new


# In[12]:


def stemming(file_tokenized_list):
    file_tokenized_list_new=[]
    stemmer=PorterStemmer()
    for token in file_tokenized_list:
        file_tokenized_list_new.append(stemmer.stem(token))
    return file_tokenized_list_new


# In[15]:


#path to get all files
path=os.getcwd()+'/stories'

#Create a dictionary for all the tokenised results
file_tokenized_dic=dict()

count_tokens=0
count_types=0

for file in os.listdir(path):
    #File load
    print("Loading File",file)
    file_payload=dataLoad(path+'/'+file)
        
    #Preprocessing begins
    #1. Remove accented characters if any
    file_payload=accentedCharacters(file_payload)
    
    #2. Expand contractions
    file_payload=expandContractions(file_payload)
    
    #3. Remove special characters 
    file_payload=specialCharactersRemoval(file_payload)
    
    #4. Lowercase all data
    file_payload=toLowerCase(file_payload)
    
    #5. Tokenization
    file_tokenized_list=tokenizer(file_payload)
    #print(file_tokenized_list)
    
    #6. Number to words-currently not doing it
    file_tokenized_list=numberToWords(file_tokenized_list)
    #print(len(file_tokenized_list))
    
    #7. Removal of stop words
    file_tokenized_list=stopWordRemoval(file_tokenized_list)
    #print(file_tokenized_list)
    #print(len(file_tokenized_list))
    
    #8. Single Character removal
    file_tokenized_list=singleCharRemoval(file_tokenized_list)
    
    #9.Lammetization
    file_tokenized_list=lemmatization(file_tokenized_list)
    #print(file_tokenized_list)
    #print(len(file_tokenized_list))
    
    #10. Stemming
    file_tokenized_list=stemming(file_tokenized_list)
    #print(file_tokenized_list)
    #print(len(file_tokenized_list))
        
    #Add entry to the dictionary
    file_tokenized_dic[file]=file_tokenized_list
    
    count_tokens+=len(file_tokenized_list)
    count_types+=len(set(file_tokenized_list))
    


#Store the dictionary in a file
#print(file_tokenized_dic)
        
with open(os.getcwd()+"/stories_processed"+"/preprocessed_dictionary", 'wb') as f:
            pickle.dump(file_tokenized_dic, f)
        
print("No of tokens",count_tokens)
print("No of types",count_types)


# In[14]:


#check pickled data
with open(os.getcwd()+"/"+"stories_processed"+"/"+"preprocessed_dictionary_small", 'rb') as f:
    new_dict = pickle.load(f)
#print(new_dict)

