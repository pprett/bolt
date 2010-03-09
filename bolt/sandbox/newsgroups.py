import bolt
import nltk
import string
import numpy as np
import sys


stopwords = nltk.corpus.stopwords.words(fileids="english")
stopwords.extend(string.punctuation)
stopwords = set(stopwords)

news = nltk.corpus.news

fd = nltk.FreqDist((word.lower() for word in news.words(fileids = news.fileids()) if word not in stopwords))

MIN_TF = 5
voc = set([w for w,freq in fd.items() if freq > MIN_TF])

voc = sorted(voc)
voc = dict(((w,i) for i,w in enumerate(voc)))
dim = len(voc)

docs = []
labels = []

cats = dict(((c,i) for i,c in enumerate(news.categories())))

for fid in news.fileids():
    c = fid[:fid.find("/")]
    labels.append(cats[c])
    fd = nltk.FreqDist((word.lower() for word in news.words(fileids=fid)))
    doc = np.array([(voc[w],float(tf)) for w,tf in fd.items() if w in voc], dtype=bolt.sparsedtype)
    doc['f1'] /= np.linalg.norm(doc['f1'])
    docs.append(doc)

labels = np.array(labels)
docs = np.array(docs,dtype=object)

data = zip(docs,labels)
np.random.shuffle(data)

cutoff = int(len(data) * 0.6)
train = data[:cutoff]
test = data[cutoff:]

itrain,ltrain = zip(*train)
itest,ltest = zip(*test)

itrain,ltrain = np.array(itrain,dtype=np.object),np.array(ltrain,dtype=np.float32)
itest,ltest = np.array(itest,dtype=np.object), np.array(ltest,dtype=np.float32)

print "%d training docs." % len(itrain)
print "%d test docs. " % len(itest)

f = open("news_train.npy",'w+b')
try:
    np.save(f,itrain)
    np.save(f,ltrain)
    np.save(f,dim)
finally:
    f.close()

f = open("news_test.npy",'w+b')
try:
    np.save(f,itest)
    np.save(f,ltest)
    np.save(f,dim)
finally:
    f.close()
