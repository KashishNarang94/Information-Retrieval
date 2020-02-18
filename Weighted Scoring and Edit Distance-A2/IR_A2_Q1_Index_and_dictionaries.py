#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle


# In[2]:


def invertedIndexCreation(file_tokenised_dic):
    #print(file_tokenized_dic)
    
    
    #Create a dictionary of {docID:docName}
    doc_name_id=dict()
    id_counter=0
    
    #Copy of master dictinary of articles using docId instead docName used in filetokenised_dic
    docID_tokenised_dic=dict()
    
    #Format is : {term:{docID:term_freq}}
    inverted_index=dict()

    #Format is : {term:doc_freq} If required we do it later, as we can directly have doc_freq by len(inverted_index[term].keys())
    #term_doc_freq=dict()
    
    for doc_name,token_list in file_tokenised_dic.items():
        print("Processing doc---------",doc_name," giving ID",id_counter)
        
        #1. Adding entry to document name and id mapping
        doc_name_id[id_counter]=doc_name
        
        #2. Copy of master dictinary of articles using 
        docID_tokenised_dic[id_counter]=token_list
        
        #2. Inverted index creation for articles
        for token in token_list:
            if token not in inverted_index.keys():
                inverted_index[token]={id_counter:1}
            elif token in inverted_index.keys() and id_counter not in inverted_index[token].keys():
                #Add docID to posting list with term frquency 1
                inverted_index[token][id_counter]=1
            elif token in inverted_index.keys() and id_counter in inverted_index[token].keys():
                #Increase the term frequency by one
                inverted_index[token][id_counter]+=1
                
            #Sorting the posting list dictionary
            inverted_index[token]=dict(sorted(inverted_index[token].items()))
        
        id_counter+=1  #Increase docID for next doc_name
        
    return doc_name_id,docID_tokenised_dic,inverted_index
            


# In[3]:


def printDictionary(dic):
    for key,value in dic.items():
        print(key,value)


# In[4]:


#------------------------For Data Files------------------------

path=os.getcwd()+'/stories_processed'

#Load tokenised word dictionaries
with open(path+"/"+"preprocessed_dictionary", 'rb') as f:
    file_tokenized_dic = pickle.load(f)
        
doc_name_id,docID_tokenised_dic,inverted_index=invertedIndexCreation(file_tokenized_dic)

#Store docID-docName Mapping
with open(path+"/"+"DocName_DocId_Mapping", 'wb') as f1:
            pickle.dump(doc_name_id, f1)
        
#Store master vocab using docID
with open(path+"/"+"dictionary", 'wb') as f1:
            pickle.dump(docID_tokenised_dic, f1)

#Store the dictionary
with open(path+"/"+"inverted_index", 'wb') as f:
            pickle.dump(inverted_index, f)

print("Inverted index length",len(inverted_index))
print("Inverted index stored as file named 'inverted_index'")


# In[5]:


printDictionary(inverted_index)


# In[15]:


printDictionary(docID_tokenised_dic)


# In[6]:


printDictionary(doc_name_id)


# In[16]:


#-----------------------For titles---------------------------

path=os.getcwd()+'/stories_processed'

#Load tokenised word dictionaries
with open(path+"/"+"Tpreprocessed_dictionary_small", 'rb') as f:
    file_tokenized_dic = pickle.load(f)
        
Tdoc_name_id,TdocID_tokenised_dic,Tinverted_index=invertedIndexCreation(file_tokenized_dic)

#Store docID-docName Mapping
with open(path+"/"+"TDocName_DocId_Mapping_small", 'wb') as f1:
            pickle.dump(Tdoc_name_id, f1)
        
#Store master vocab using docID
with open(path+"/"+"Tdictionary_small", 'wb') as f1:
            pickle.dump(TdocID_tokenised_dic, f1)

#Store the dictionary
with open(path+"/"+"Tinverted_index_small", 'wb') as f:
            pickle.dump(Tinverted_index, f)


# In[17]:


printDictionary(Tinverted_index)

