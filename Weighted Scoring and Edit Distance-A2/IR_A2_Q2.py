#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np


# In[4]:


def loadEnglishDictionary():
    path = os.getcwd()
    
    english_vocab=[]
    with open(path+"/"+"english.txt") as f:
        for line in f:
            english_vocab.append(line.rstrip('\n'))
    return english_vocab
        


# In[5]:


def inputQuery():
    query = input("Enter your query: ") 
    query = query.split()
    return query


# In[6]:


def notInDictionary(query,dictionary):
    not_in_dictionary=list(set(query)-set(dictionary))
    return not_in_dictionary


# In[7]:


def printDictionary(dic,message):
    print(message)
    print("ID",":","Score")
    for key,value in dic.items():
        print(key,":",value)


# In[8]:


def editDistance(str1,str2):
    cost_delete=1
    cost_insert=2
    cost_replace=2
   
    edit_matrix=[[0 for x in range(len(str2)+1)] for x in range(len(str1)+1)]
    
    #Initialize first row and column
    for i in range(1,len(str1)+1):
        edit_matrix[i][0] = i * cost_delete
    for j in range(1,len(str2)+1):
        edit_matrix[0][j]=j * cost_insert
    
    #A well known algorithm : https://en.wikipedia.org/wiki/Levenshtein_distance
    for j in range(1,len(str2)+1):
        for i in range(1,len(str1)+1):
            if str1[i-1]==str2[j-1]:
                cost=0
            else:
                cost=cost_replace
            edit_matrix[i][j] = min(edit_matrix[i-1][j] + cost_delete,edit_matrix[i][j-1] + cost_insert,edit_matrix[i-1][j-1] + cost)
    
    #for i in range(len(str1)+1):
    #    print(edit_matrix[i])

    return edit_matrix[len(str1)][len(str2)]


# In[9]:


def topK(q_word, dic):
    edit_dic=dict()
    no_words=10
    
    for dic_word in dic:
        edit_cost=editDistance(q_word,dic_word)
        edit_dic[dic_word]=edit_cost
    
    #Sort docment ID based on edit cost
    edit_dic={ky: v for ky, v in sorted(edit_dic.items(), key=lambda item: item[1])}
    
    printDictionary(edit_dic,"Increasing sorted edit dictionary")
    
    #top K docID list 
    topk=[]
    
    #Putting dic_words in topk list
    for i in range(0,no_words):
        topk.append(list(edit_dic.keys())[i])
    return topk
    


# In[10]:


def processQuery(not_in_dictionary,dictionary):
    print("Query words not in dictionary",not_in_dictionary)
    for q_word in not_in_dictionary:
        suggestions=topK(q_word,dictionary)
        print("Top 10 suggested words for ",q_word,"are")
        for i in range(len(suggestions)):
            print(suggestions[i])


# In[11]:


dictionary=loadEnglishDictionary()
query=inputQuery()
not_in_dictionary=notInDictionary(query,dictionary)
processQuery(not_in_dictionary,dictionary)

