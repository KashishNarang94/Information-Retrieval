#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pickle
import math

import import_ipynb
import IR_Helper_Functions as ir_helper


# In[38]:


def loadDictionariesData():
    path=os.getcwd()+'/stories_processed'
    
    #---------------------Data files---------------------------
    #Load master dictionary {DocID:[tokens]}
    with open(path+"/"+"dictionary", 'rb') as f:
        docID_tokenised_dic = pickle.load(f)
    
    #Load docName-DocID mapping
    with open(path+"/"+"DocName_DocId_Mapping", 'rb') as f:
        doc_name_id = pickle.load(f)
    
    #Load inverted index
    with open(path+"/"+"inverted_index", 'rb') as f1:
        inverted_index = pickle.load(f1)
    
    return doc_name_id,docID_tokenised_dic,inverted_index


# In[3]:


def loadDictionariesTitle():
    path=os.getcwd()+'/stories_processed'
    
    #---------------------Title files---------------------------
    #Load master dictionary {DocID:[tokens]}
    with open(path+"/"+"Tdictionary_small", 'rb') as f:
        TdocID_tokenised_dic = pickle.load(f)
    
    #Load docName-DocID mapping
    with open(path+"/"+"TDocName_DocId_Mapping_small", 'rb') as f:
        Tdoc_name_id = pickle.load(f)
    
    #Load inverted index
    with open(path+"/"+"Tinverted_index_small", 'rb') as f1:
        Tinverted_index = pickle.load(f1)
    
    return Tdoc_name_id,TdocID_tokenised_dic,Tinverted_index


# In[9]:


def inputQuery():
    query = input("Enter your query: ") 
    k=input("Enter number of records you want me to fetch")
    
    print("Enter the calculation method you want to use: 1. Jaccard  2. TF-IDf Matching  3. Cosine Similarity ")
    cal_method=input()
    
    if cal_method!=str(1):
        print("Enter the type code for Tf calulation: 1. Raw Tf  2. Normalised TF  3. Log based TF 4. Double Norm")
        type_code=input()
        return query,k,cal_method,type_code
    return query,k,cal_method,0


# In[42]:


def printDictionary(dic,message):
    print(message,"with length ",len(dic))
    print("ID",":","Score")
    for key,value in dic.items():
        print(key,":",value)


# In[11]:


def convertAndPrintDocName(results_docIds,doc_name_id):
    no_records_retrieved=len(results_docIds)
    if no_records_retrieved>0:
        print("Top",no_records_retrieved,"documents are:")
        for docId in results_docIds:
            print(doc_name_id[docId])
    else:
        print("No matching record found")


# In[63]:


def statistics(dic,type_code,cal_method):
    
    #Count non-zero
    count_non_zero=0
    for i in dic.values():
        if i>0.0:
            count_non_zero+=1
     
    print("--------------Statistics of the result are ----------- ")
    print("Method for Ranking : ",cal_method,"and method for TF",type_code)
    print("1. Number of dictionary items : ",len(dic))
    print("2. Highest Score : ",max(dic.values()))
    print("3. Lowest score : ",min(dic.values()))
    print("4. Number of records having score >0 :",count_non_zero)
    print("5. % of such records : ",count_non_zero/467)


# In[54]:


def jaccardRanking(query_pr,docID_tokenised_dic,no_records):
    jaccard_rank_dic=dict()
    
    #Make set of query
    query_pr=set(query_pr)
    
    #For each document
    for docId,tokens in docID_tokenised_dic.items():
        tokens=set(tokens)
        intersect_count=len(query_pr & tokens)
        jaccard_coff=intersect_count/(len(query_pr)+len(tokens)-intersect_count)
        jaccard_rank_dic[docId]=jaccard_coff
    
    #Sort docment ID based on jaccard coff
    jaccard_rank_dic={ky: v for ky, v in sorted(jaccard_rank_dic.items(), key=lambda item: item[1], reverse=True)}
    
    #printDictionary(jaccard_rank_dic,"Decreasing sorted jaccard dictionary")
    statistics(jaccard_rank_dic,0,1)
    
    #top K docID list 
    topk=[]
    
    #Setting value of no_records
    if len(jaccard_rank_dic.keys())==0:
        return topk   #return empty list
    elif no_records<len(jaccard_rank_dic.keys()):
        no_records=no_records
    elif no_records>=len(jaccard_rank_dic.keys()):
        no_records=len(jaccard_rank_dic.keys())
    
    #Putting docId in topk list
    for i in range(0,no_records):
        if jaccard_rank_dic[list(jaccard_rank_dic.keys())[i]]!=0.0:
            topk.append(list(jaccard_rank_dic.keys())[i])
    return topk
    


# In[28]:


def maximumTfInDoc(docId,inverted_index):
    max_tf_term_count=0
    t=""
    for term,value_dic in inverted_index.items():
        for docId_,tf in value_dic.items():
            if docId==docId_ and tf>max_tf_term_count:
                max_tf_term_count=tf
                t=term
                
    #print(t,max_tf_term_count)
    return max_tf_term_count
    


# In[53]:


def tfCalci(raw_tf,no_words_doc,max_tf_term_count,type_code):
    if type_code==1:          #raw tf
        return raw_tf
    elif type_code==2:        #tf_normalised by N
        tf=raw_tf/no_words_doc
        return tf
    elif type_code==3:         #log based
        tf_log=1+math.log(raw_tf)
        return tf_log
    elif type_code==4:
        tf_doublenorm=0.5+(0.5*(raw_tf/max_tf_term_count))
        return tf_doublenorm


# In[30]:


def idfCalci(raw_df,no_docs):
    if raw_df==0:
        return 0
    else:
        idf=math.log(no_docs/raw_df)
        return idf


# In[52]:


def tfIdfMatchRanking(query_pr,docID_tokenised_dic,inverted_index,no_records,type_code):
    tfidf_rank_dic=dict()
    no_docs=len(docID_tokenised_dic.keys())
    
    for term in query_pr:
        if term in inverted_index.keys():
            postingdata=inverted_index[term]
            #print(term,postingdata)
        
            #IDf calculation
            raw_df=len(postingdata)
            idf=idfCalci(raw_df,no_docs)
            #print("No of docs",no_docs)
            #print("DF is",raw_df)
            #print("IDF is",idf)
        
            for docId,raw_tf in postingdata.items():
                no_words_doc=len(docID_tokenised_dic[docId])
                #print("No of words in docID",docId,"is",no_words_doc)
                max_tf_term_count=maximumTfInDoc(docId,inverted_index)
                
                #Calculate Tf
                tf=tfCalci(raw_tf,no_words_doc,max_tf_term_count,type_code)
                #print("TF is",tf)
                
                #Calculate Tf-Idf
                tfidf=tf*idf
                #print("TF-IDF score is",tfidf,"for docId",docId)
            
                if docId in tfidf_rank_dic.keys():
                    tfidf_rank_dic[docId]+=tfidf
                else:
                    tfidf_rank_dic[docId]=tfidf
    
    #Sort docment ID based on tfidf matching score
    tfidf_rank_dic={ky: v for ky, v in sorted(tfidf_rank_dic.items(), key=lambda item: item[1], reverse=True)}
    
    #printDictionary(tfidf_rank_dic,"Decreasing sorted tfidf dictionary")
    statistics(tfidf_rank_dic,type_code,2)
    
    #top K docID list 
    topk=[]
    
    #Setting value of no_records
    if len(tfidf_rank_dic.keys())==0:
        return topk   #return empty list
    elif no_records<len(tfidf_rank_dic.keys()):
        no_records=no_records
    elif no_records>=len(tfidf_rank_dic.keys()):
        no_records=len(tfidf_rank_dic.keys())
    
    #Putting docId in topk list
    for i in range(0,no_records):
        if tfidf_rank_dic[list(tfidf_rank_dic.keys())[i]]!=0.0:
            topk.append(list(tfidf_rank_dic.keys())[i])
    return topk   
        


# In[32]:


def queryMasterDictioanry(query_pr):
    #{term : tf}
    query_term_tf=dict()
    
    for term in query_pr:
        if term not in query_term_tf.keys():
            query_term_tf[term]=1
        elif term in query_term_tf.keys():
            query_term_tf[term]+=1
    return query_term_tf


# In[33]:


def queryTfidfVector(query_pr,docID_tokenised_dic,inverted_index,type_code):
    no_docs=len(docID_tokenised_dic.keys())
    
    #dictionary for query tfidf storage {term:tdidf}
    query_vector_dic=dict()
    
    #create amster dictioary for query
    query_term_tf=queryMasterDictioanry(query_pr)
    #print("Query term - tf ",query_term_tf)
    
    no_words_doc_q=len(query_pr)
    max_tf_term_count=max(query_term_tf.values())
    
    for term_q,raw_tf_q in query_term_tf.items():
        if term_q in inverted_index.keys():
            #IDf calculation
            raw_df=len(inverted_index[term_q])
            idf=idfCalci(raw_df,no_docs)

            #Calculate Tf
            tf_q=tfCalci(raw_tf_q,no_words_doc_q,max_tf_term_count,type_code)
            #print("TF is",tf)
                
            #Calculate Tf-Idf
            tfidf_q=tf_q*idf
            #print("Query TF-IDF score is",tfidf_q,"for term",term_q)
            
            if term_q not in query_vector_dic.keys():
                query_vector_dic[term_q]=tfidf_q
        else:
            if term_q not in query_vector_dic.keys():
                query_vector_dic[term_q]=0
    
    return query_vector_dic


# In[34]:


def documentVector(query_pr,docID_tokenised_dic,inverted_index,type_code):
    no_docs=len(docID_tokenised_dic.keys())
    
    #dictionary of all docs terms as tfidf {docID:{term:tfidf},docID:{term:tfidf}
    docs_vector_dic=dict()
    
    for docId,tokens in docID_tokenised_dic.items():
        docs_vector_dic[docId]=dict()
        for term in query_pr:
            if term in tokens:
                postingdata=inverted_index[term]
                #print(docId,term,postingdata)
                
                #IDf calculation
                raw_df=len(postingdata)
                idf=idfCalci(raw_df,no_docs)
                
                #creating tfidf vector for documents
                for docId_p,raw_tf in postingdata.items():
                    if docId==docId_p:
                        no_words_doc=len(docID_tokenised_dic[docId])
                        #print("No of words in docID",docId,"is",no_words_doc)
                        max_tf_term_count=maximumTfInDoc(docId,inverted_index)
                        
                        #Calculate Tf
                        tf=tfCalci(raw_tf,no_words_doc,max_tf_term_count,type_code)
                        #print("TF is",tf)
                
                        #Calculate Tf-Idf
                        tfidf=tf*idf
                        #print("TF-IDF score is",tfidf,"for term",term,"in docID",docId)
            
                        if term not in docs_vector_dic[docId].keys():
                            docs_vector_dic[docId][term]=tfidf
                        
            elif term not in tokens:
                #Put tfidf value as 0
                docs_vector_dic[docId][term]=0
            
    return docs_vector_dic


# In[51]:


def cosineCalci(vec1,vec2):
    num,den1,den2=0,0,0
    
    for i in range(len(vec2)):
        num+=(vec1[i]*vec2[i])
        den1+=(vec1[i]*vec1[i])
        den2+=(vec2[i]*vec2[i])
    if num!=0 and den1!=0 and den2!=0:    
        cosine_score=num/(math.sqrt(den1*den2))
    else:
        cosine_score=0
    
    return cosine_score


# In[50]:


def cosineRanking(query_pr,docID_tokenised_dic,inverted_index,no_records,type_code):
    docs_vector_dic=documentVector(query_pr,docID_tokenised_dic,inverted_index,type_code)
    query_vector_dic=queryTfidfVector(query_pr,docID_tokenised_dic,inverted_index,type_code)
    
    #print("Documents TFidf vector",docs_vector_dic)
    #print("Query tfidf vector",query_vector_dic)
    
    #Cosine similarity calculation and storage
    cosine_score_dic=dict()
    
    for docId,doc_vector in docs_vector_dic.items():
        vec1=list(doc_vector.values())
        vec2=list(query_vector_dic.values())
        
        cosine_score=cosineCalci(vec1,vec2)
        cosine_score_dic[docId]=cosine_score    
    
    #Sort docment ID based on tfidf matching score
    cosine_score_dic={ky: v for ky, v in sorted(cosine_score_dic.items(), key=lambda item: item[1], reverse=True)}
    
    #printDictionary(cosine_score_dic,"Decreasing sorted cosine dictionary with length")
    statistics(cosine_score_dic,type_code,3)
    
    #top K docID list 
    topk=[]
    
    #Setting value of no_records
    if len(cosine_score_dic.keys())==0:
        return topk   #return empty list
    elif no_records<len(cosine_score_dic.keys()):
        no_records=no_records
    elif no_records>=len(cosine_score_dic.keys()):
        no_records=len(cosine_score_dic.keys())
    
    #Putting docId in topk list
    for i in range(0,no_records):
        if cosine_score_dic[list(cosine_score_dic.keys())[i]]!=0.0:
            topk.append(list(cosine_score_dic.keys())[i])
    return topk   


# In[68]:


#--------------MAin Funtionioning-----------------

#Load Dictionaries for various usage
doc_name_id,docID_tokenised_dic,inverted_index=loadDictionariesData()
#Tdoc_name_id,TdocID_tokenised_dic,Tinverted_index=loadDictionariesTitle()

#Query input
query_raw,k,cal_method,type_code=inputQuery()
k=int(k)
cal_method=int(cal_method)
type_code=int(type_code)

#Query Preprocessing
query_pr=ir_helper.preprocess(query_raw)
print("Query token list",query_pr)

if cal_method==1:
    #Jaccard based ranking 
    result_docIds=jaccardRanking(query_pr,docID_tokenised_dic,k)
elif cal_method==2:
    #TF-IDF Matching score
    result_docIds=tfIdfMatchRanking(query_pr,docID_tokenised_dic,inverted_index,k,type_code)
elif cal_method==3:
    #Cosine Similarity
    result_docIds=cosineRanking(query_pr,docID_tokenised_dic,inverted_index,k,type_code)

#Print Results
convertAndPrintDocName(result_docIds,doc_name_id)

