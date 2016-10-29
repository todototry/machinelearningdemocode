# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:36:20 2013

@author: fandyst
"""
import numpy as np
import pylab as pl
import pandas as pd


def loadSample(filename,sep = '\t'):
    f = open(filename)
    X = []
    Y = []
    for line in f.readlines():
        l = line.split(sep)
        X.append([float(l[0]), float(l[1])])
        Y.append(float(l[2]))
    return X,Y


def pdLoadMnist(filename,sep = ','):
    
    traindata = pd.read_csv(filename)    
    minidata = traindata
    testdata = traindata.tail(100)
    
    Ylabel = minidata.iloc[:]['label']
    Ytestlabel = testdata.iloc[:]['label']
    Ytestlabel = np.asarray(Ytestlabel)
    
    X = minidata.iloc[:,1:]
    Xtest = testdata.iloc[:,1:]
    
    X = X.as_matrix()
    XT = Xtest.as_matrix()
    
    Y = -np.ones((len(Ylabel),10))
    YT = -np.ones((len(Ytestlabel),10))
    
    for i in range(len(Ylabel)):
        Y[i,Ylabel[i]] = 1.
        
    #print Ytestlabel[2]
    
    for j in range(len(Ytestlabel)):        
        YT[j,Ytestlabel[j]] = 1.
    
    return X,Y,XT,YT



class nnet:
    #layer num, hidenNodeNum , X:samples, Y:targets for each points
    def __init__(self,layernum,hidenNodeNum,x,y):
        sampleWidth = np.shape(X)[1]
        self.Layers = layernum
        self.NodesInEachLayer = hidenNodeNum
        #init the weights at range of (6./2*nodes_num_at_each_layer)**0.5
        w0_1d = np.random.normal(0,1,sampleWidth*hidenNodeNum)*((6./(2*hidenNodeNum))**0.5)
        self.W0 = w0_1d.reshape((sampleWidth,hidenNodeNum))
        
        #3d index here in numpy-->(z,x,y)  z(layer index), x(weight_index for node y), y (node index in z)
        w1 = np.random.normal(0,1,(layernum-1)*(hidenNodeNum**2))
        
        w1 = w1*((6.**0.5)/((2*hidenNodeNum)**0.5))
        self.W = np.reshape(w1,(layernum-1,hidenNodeNum,hidenNodeNum))
        
        self.NodesValue = np.zeros((layernum,hidenNodeNum))
        
        self.Errors = np.zeros((layernum,hidenNodeNum))
        
        self.X = x
        self.Y = y
        
        self.timesMax = 150
        self.eps = 0.001
        self.Xcur = 0
        
    def sigmoid(self,x):
        return 1./(1.+np.e**(-x))
    
    def sFunction(self):
        return self.sigmoid

    #calculate nodes' values in each layer
    #notice that ...................at the second to last layer. the output of node value is f(Node_j-1)
    def forwardNetwork(self,Xindex):
        self.Xcur = Xindex
        for layerindex in range(self.Layers):
            for nodeindex in range(self.NodesInEachLayer):
                if layerindex == 0 :
                    self.NodesValue[layerindex][nodeindex] = sum(self.X[Xindex] * self.W0[:,nodeindex])
                else:
                    self.NodesValue[layerindex][nodeindex] = sum(self.sFunction()(self.NodesValue[layerindex-1,:]) *  (self.W[layerindex-1,:,nodeindex]))


    #calc error for each node.
    def calcErrors(self):
        #calc the last layer.
        #yi - label.
        for i in range(self.NodesInEachLayer):
            oj = self.sFunction()(self.NodesValue[-1][i])
            self.Errors[-1][i] = (self.Y[-1][i]-oj)*((1-oj)*oj)
        #calc errors for nodes in all layers except the out layer.
        for layerindex in range(self.Layers-2,-1,-1):
            for nodeindex in range(self.NodesInEachLayer):
                #all weights of cur node, multiply the errors of  next layer. 
                Sk = sum(self.sFunction()(self.Errors[layerindex+1][:]) * self.W[layerindex,:,nodeindex])
                oj = self.sFunction()(self.NodesValue[layerindex][nodeindex])
                self.Errors[layerindex][nodeindex] = Sk * (1-oj) *oj


    def adjust(self):
        #error back propagation. --> compute error for each node at each layer
        # done !
        #adjust the weights for each node.
        # in all W0 , calc the new weight of each input feature's wi for each node 
        for featureindex in range(self.X.shape[1]):
            for nodeindex in range(self.NodesInEachLayer):
                self.W0[featureindex][nodeindex] += self.Errors[0][nodeindex]*self.sFunction()(self.NodesValue[0][nodeindex])
        #for the middle hiden layers.
        for formerlayerindex in range(1,self.Layers):
            for nodeindex in range(self.NodesInEachLayer):
                for weightindex in range(self.NodesInEachLayer):
                    self.W[formerlayerindex-1][weightindex][nodeindex] += self.Errors[formerlayerindex][nodeindex]*self.sFunction()(self.NodesValue[formerlayerindex][nodeindex])        


    def BP_train(self):
        #train by which sample ?
        deltaS = 0.
        times = 0
        for i in xrange(len(self.X)):
            #when do we stop train ?
            #the stop condition is determined by the adjusting times and the value changes of each adjusting step.
            while times < self.timesMax:                
                self.forwardNetwork(i)
                self.calcErrors()
                deltaS = self.adjust()
                times+=1


    def test(self,xtest,ytest):
        self.X  = xtest
        total = len(xtest)
        correct = 0
        result = 0
        for i in range(total):
            self.forwardNetwork(i)
            result = self.sFunction()(self.NodesValue[self.Layers-1][:])
            if np.argmax(result) == np.argmax(ytest[i]):
            #indexN = np.argmax(ytest[i])
            #print result[:],indexN
                correct += 1
        return correct,total



if __name__ == "__main__":
    X,Y,XT,YT = pdLoadMnist("/home/fandyst/train.csv")
    net = nnet(3,10,X,Y)
    print net.NodesValue
    #print net.Errors
    #net.forwardNetwork(0)
    net.BP_train()
    #print net.W0
    #print net.Errors
    print net.test(XT,YT)







