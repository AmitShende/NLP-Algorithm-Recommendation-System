from flask import Flask,render_template,request
import pandas as pd, numpy as np, os, re, operator, csv, string
from operator import itemgetter
from collections import Counter, defaultdict

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 #.download('words')
words = set(nltk.corpus.words.words())

from gensim.models.tfidfmodel import TfidfModel
from gensim import similarities, models, corpora, utils
from gensim.test.utils import datapath, get_tmpfile

os.chdir('D:\\Data_science\\Projects\\Excelr_project\\project1_NLP\\output\\Final_files\\Step_VI_Deployment')
#============================================
path = os.getcwd()
df = pd.read_csv('df3.csv') 
# Checking any empty rows or Na values in rows... if avalable it will be deleted
# keyword column is created to store the text from NLP cleaned question and answer columns. 
df.isnull().sum()
df = df[df['keywords'].notna()]   

# Resetting Index
df.reset_index

# Processing Keywords: tokenising the words for creating bag of words
docs = df['keywords'].tolist()
processed_docs = [word_tokenize(doc.lower()) for doc in docs] #tokenize and lowercase to generate bag of words

#=========================================================
# Step I-------creating similarity model for the data file   
#=========================================================

# Creating and saving dictionary        
dictionary = corpora.Dictionary(processed_docs) # create a dictionary of words from our keywords
dictionary.save(path+'dim_items_terms.dict') #saved

# Creating and saving corpus
corpus = [dictionary.doc2bow(doc) for doc in processed_docs] #create corpus where the corpus is a bag of words for each document
corpora.MmCorpus.serialize(path+'dim_items_terms.mm', corpus) #saved

#---Creating TFIDF MATRIX  
          
tfidf = TfidfModel(corpus) # # step 1 -- initialize a model i.e. create tfidf model of the corpus (train the transformation model)
tfidfmodelsave=tfidf.save(path+'dim_items_terms.tfidf') #saved
tfidf_corpus=tfidf[corpus] 

#checking allocations of weights from tfidf matrix------
sorted_tfidf_weights = sorted(tfidf[corpus[0]], key=lambda w: w[1], reverse=True)
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)

#finding cosine similarity from tfidf matrix------
sims = Similarity('path1',tfidf[corpus], num_features=len(dictionary))
sims.save(path+'_saved_sims.similarity')

app=Flask(__name__)

@app.route("/")
def projectname():
    return render_template("intro.html")

@app.route("/intro")
def intro():
    return render_template("home.html",methods=["POST"])

@app.route("/",methods=["POST"])
def acceptinput():
    Top_N_Q_and_A=5
    keyword_rec = request.form["Enter keyword or phrase:"]    
    keywords_rec=keyword_rec
    keywords_rec = keywords_rec.replace('.',' ') 
    keywords_rec=re.sub("<!--?.*?-->","",keywords_rec)
    keywords_rec=re.sub("(\\d|\\W)+"," ",keywords_rec)
    keywords_rec = re.sub(r'\s+',' ',re.sub(r'[^\w \s]','',keywords_rec) ).lower()
    keywords_rec = word_tokenize(keywords_rec) # tokenize for generating bag of words
    keywords_rec= [w for w in keywords_rec if w not in set(stopwords.words('english'))]
        
    # After cleaning the given question keyword using text nlp, if keyword returns empty then given keywords will be assigned/used directly for analysis 
    if len(keywords_rec)==0:
       keywords_rec=word_tokenize(keyword_rec)
    # creating model for the keyword 
    query_doc_bow = dictionary.doc2bow(keywords_rec) # get a bag of words from the query_doc
    query_doc_tfidf = tfidf[query_doc_bow] #convert the regular bag of words model to a tf-idf model of the keywords and it's tf-idf value for the question and answer
              
    similarity_array = sims[query_doc_tfidf] # get the array of similarity values between our new keywords and every other keywords. 
                        
    similarity_series = pd.Series(similarity_array.tolist(), index=df.Q_and_A.values) #Convert to a Series
    
    Top_Q_and_A1=similarity_series.sort_values(ascending=False)
    Top_Q_and_A=Top_Q_and_A1.head(5)
    #Top_Q_and_A = similarity_series.sort_values(ascending=False)[:Top_N_Q_and_A] #get the top matching results, 
    #return render_template("2ndpage.html",Top_N_Q_and_A=100)   
    QA=[Top_Q_and_A.index[0], Top_Q_and_A.index[1], Top_Q_and_A.index[2], Top_Q_and_A.index[3], Top_Q_and_A.index[4]]
    Score=[Top_Q_and_A[0], Top_Q_and_A[1], Top_Q_and_A[2], Top_Q_and_A[3], Top_Q_and_A[4]]
    
    
    QA1_Q= QA[0].split('\n',1)[0]  
    QA1_A= QA[0].split('\n',1)[1]
    Simalarity_Score1=round(Score[0]*100,2)

    QA2_Q= QA[1].split('\n',1)[0]  
    QA2_A= QA[1].split('\n',1)[1]
    Simalarity_Score2=round(Score[1]*100,2)
        
    QA3_Q= QA[2].split('\n',1)[0]    
    QA3_A= QA[2].split('\n',1)[1]
    Simalarity_Score3=round(Score[2]*100,2)
        
    QA4_Q= QA[3].split('\n',1)[0]
    QA4_A= QA[3].split('\n',1)[1] 
    Simalarity_Score4=round(Score[3]*100,2)
        
    QA5_Q= QA[4].split('\n',1)[0] 
    QA5_A= QA[4].split('\n',1)[1]
    Simalarity_Score5=round(Score[4]*100,2)
    return render_template("results.html",K1=keyword_rec, a=QA1_Q, b=QA1_A, c=Simalarity_Score1, d=QA2_Q, e=QA2_A, f=Simalarity_Score2, g=QA3_Q, h=QA3_A, i=Simalarity_Score3, j=QA4_Q, k=QA4_A, l=Simalarity_Score4, m=QA5_Q, n=QA5_A, o=Simalarity_Score5)
   
if __name__=="__main__": 
 app.run() 
    
