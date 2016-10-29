# coding: utf-8
persontype = dtype({'names':['name','age','weight'],'formats':['S32','i','f']},align=True)
get_ipython().magic(u'pinfo dtype')
import numpy as np
datafile = open("datingTestSet2.txt")
sampleLen = len(datafile.readlines())
sampleLen
dataset = zeros((sampleLen,4))
dataset.dtype

def loadSample(filename):
    datafile = open(filename)
    sampleLen = len(datafile.readlines())
    dataset = zeros((sampleLen,4))
    datafile = open("datingTestSet2.txt")
    for index,data in enumerate(datafile.readlines()):
        data = data.strip()
        sample = data.split('\t')
        dataset[index,:] = sample[:]
    return dataset
dataset = loadSample("datingTestSet2.txt")
dataset[1,0)
dataset[0,0:]
get_ipython().system(u'ls -F --color -F -o --color ')


import operator
def classify(x,dataset,samplesize,k):
    xxx = tile(x,(samplesize,1))
    diff = dataset[:,0:3]-xxx
    dis = diff**2
    sqDis = sum(dis,axis = 1)
    distances = sqDis**0.5
    sortedDistancesindex = distances.argsort()
    classCount = {}
    for i in range(k):
        label = dataset[sortedDistancesindex[k],3]
        classCount[label] = classCount.get(label,0)+1
    sortedClass = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sortedClass[0][0]
testdata = dataset[900:1000,:]

traindata = dataset[0:900,:]

        
errorCount = 0
for index,d in enumerate(testdata[:,0:3]):
    c = classify(d,traindata,900,5)
    if int(testdata[index,3]) != int(c) :
        errorCount += 1
errorCount


24/100.


def normdata(dataset):
    datasize = dataset.shape[0]
    mins = dataset.min(0)
    maxs = dataset.max(0)
    ranges = maxs - mins
    rangesmat = tile(ranges,(datasize,1))
    minsmat = tile(mins,(datasize,1))
    return (dataset-minsmat)/rangesmat

    
normtestdata = normdata(testdata)
normtraindata = normdata(traindata)

def testclassify(testdata,traindata):
    errorCount = 0
    for index,d in enumerate(testdata[:,0:3]):
        c = classify(d,traindata,900,5)
        if int(testdata[index,3]) != int(c) :
            errorCount += 1
    return errorCount

testclassify(normtestdata,normtraindata)

# after data normilization , the prediction success rate up
7/100.
