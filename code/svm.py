import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def loadSample(filename,sep = '\t'):
    f = open(filename)
    X = []
    Y = []
    for line in f.readlines():
        l = line.split(sep)
        X.append([float(l[0]), float(l[1])])
        Y.append(float(l[2]))
    return X,Y



class smoObj:
    def __init__(self,datalist,labelslist,C,toler):
        self.X = np.mat(datalist)
        self.Y = np.mat(labelslist).transpose()
        self.C = C
        self.toler = toler
        self.n = np.shape(self.X)[0]
        self.alphas = np.mat(np.zeros((self.n,1)))
        self.b = 0.
        self.eCache = np.mat(np.zeros((self.n,2)))
        #calc and record x = Yi(W*Xi+b).  the KKT of each sample Xi, so that the selection of alpha_i will be easy to find.
        self.yiwxib = np.zeros((self.n))
        # to record |Ei-Ej|
        self.deltaE = np.zeros((self.n))
        # whether the 


def Gxi(so,i):
    return float(np.multiply(so.alphas,so.Y).T*(so.X*so.X[i,:].T))+so.b


def Ei(gxi,yi):
    ei = gxi-float(yi)
    return ei

def updateECachek(so, k):#after any alpha has changed update the new value in the cache
    Ek = Ei(Gxi(so,k), so.Y[k])
    so.eCache[k] = [1,Ek]


def calcWs(alphas,dataArr,classLabels):
    X = dataArr; 
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*classLabels[i],X[i,:].T)
    return w


def findSupportVectors(so):
    for i in range(so.X.shape[0]):
        so.yiwxib[i] = Gxi(so,i)*so.Y[i]
    indices = [i for i in so.yiwxib if so.yiwxib[i] == 1.]
    return indices


def selectI(so):
    for i in range(so.X.shape[0]):
        so.yiwxib[i] = Gxi(so,i)*so.Y[i]
    xindex = so.yiwxib.argsort()
    minindex = 0
    if minindex[0] == 1:
        minindex = xindex[-1]
    return minindex



def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

def selectJ(i, oS, ei):              #this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    oS.eCache[i,0] = 1
    oS.eCache[i,1] = ei            #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue     #don't calc for i, waste of time
            ek = Ei(Gxi(oS,k), so.Y[k])
            deltaE = abs(ei - ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; 
        return maxK
    else:                          #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.n)
    return j


#there are 2 cases to handle the Jth sample's selection.
#general case:     #find the biggest |Ei-Ej|

#special case:    
#step 1. use support vectors one by one as the alpha2. eg.the KKT(2nd condition)
#step 2. use other samples except support vectors.
#case 1: the general way
def selectJ_general(so,i,ei):    
    for j in range(so.X.shape[0]):
        gj = Gxi(so,j)
        ej = Ei(gj,so.Y[j])
        so.deltaE[j] = np.abs(ej-ei)
    eindex = so.deltaE.argsort()
    return eindex[-1]



#case 2: the special way
def selectJ_special_1(so):
    #step 1:
    indices = findSupportVectors(so)
    return indices    



#case 2: the special way
    #step 2
def selectJ_special_2(so,supportvectorsindices):
    #step 2:
    indices = set(range(so.Y.shape[0])) - set(supportvectorsindices)
    return indices



def Eta(oS,i,j):
    eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
    return eta



def clipAlpha(alpha,H,L):
    if alpha > H:
        return H
    elif alpha < L:
        return L
    else:
        return alpha



def updateAlphaJ(so,i,j,H,L):
    so.alphas[j] -= so.Y[j]*(Ei(Gxi(so,i),so.Y[i])-Ei(Gxi(so,j),so.Y[j]))/Eta(so,i,j)
    return so.alphas[j]



def updateAlphaI(so,i,j,alphaj_old):
    so.alphas[i] += so.Y[j]*so.Y[i]*(alphaj_old-so.alphas[i])
    return so.alphas[i]



def calcB1(so,i,j,alphai_old,alphasj_old):
    return so.b - Ei(Gxi(so,i),so.Y[i]) - so.Y[i]*(so.alphas[i]-alphai_old)*so.X[i,:]*so.X[i,:].T - so.Y[j]*(so.alphas[j]-alphasj_old)*so.X[i,:]*so.X[j,:].T



def calcB2(so,i,j,alphai_old,alphaj_old):
    return so.b - Ei(Gxi(so,j),so.Y[j]) - so.Y[i]*(so.alphas[i]-alphai_old)*so.X[i,:]*so.X[j,:].T - so.Y[j]*(so.alphas[j]-alphaj_old)*so.X[j,:]*so.X[j,:].T


def calcL_H(so,i,j):
    L,H = 0.,0.
    if so.Y[i] != so.Y[j]:
        L = max(0,so.alphas[j]-so.alphas[i])
        H = min(so.C,so.C+so.alphas[j]-so.alphas[i])
    else:
        L = max(0,so.alphas[j]+so.alphas[i]-so.C)
        H = min(so.C,so.alphas[j]+so.alphas[i])
    return L,H



#adjust (i1, i2)
def takeStep(so,i):
    # select j-----------------------------
    yi = so.Y[i]
    ei = Ei(Gxi(so,i),yi)
    alphai = so.alphas[i]
    if ((so.Y[i]*ei < -so.toler) and (so.alphas[i] < so.C)) or ((so.Y[i]*ei > so.toler) and (so.alphas[i] > 0)):
        j = selectJ(i, so, ei) #this has been changed from selectJrand
        alphaj = so.alphas[j]
        yj = so.Y[j]
        #To speed up. check if it is in the eCache for less computation.#################
        ej = Ei(Gxi(so,j),yj)
        s = yi*yj
        L,H = calcL_H(so,i,j)
        if L == H: return 0
        eta = Eta(so,i,j)
        if(eta < 0):
            alphajnew = alphaj - yj*(ei-ej)/eta
            if alphajnew < L:
                alphajnew = L
            elif alphajnew > H:
                alphajnew = H
        else:
            print "eta >= 0"
            return 0
        if (abs(alphajnew - alphaj) < 0.0001):
            print "j not moving enough"
            return 0
        #update i by the same amount as j  #the update is in the oppostie direction
        #see the box constrains for a1 and a2. now we get a2_new, so we can use the fucktion to calc a1_new. 
        so.alphas[i] += so.Y[j]*so.Y[i]*(alphaj - alphajnew)
        so.alphas[j] = alphajnew
        b1 = calcB1(so,i,j,alphai,alphaj)
        #so.b - ei- so.Y[i]*(so.alphas[i]-alphai)*so.X[i,:]*so.X[i,:].T - so.Y[j]*(so.alphas[j]-alphaj)*so.X[i,:]*so.X[j,:].T
        b2 = calcB2(so,i,j,alphai,alphaj)
        #so.b - ej- so.Y[i]*(so.alphas[i]-alphai)*so.X[i,:]*so.X[j,:].T - so.Y[j]*(so.alphas[j]-alphaj)*so.X[j,:]*so.X[j,:].T
        if (0 < so.alphas[i]) and (so.C > so.alphas[i]):
            so.b = b1
        elif (0 < so.alphas[j]) and (so.C > so.alphas[j]):
            so.b = b2
        else:
            so.b = (b1 + b2)/2.0
        #update the eCache, which can be used to speed up the ei and ej computation.(see lines in the 7th 8th line in this function)
        updateECachek(so, i)
        updateECachek(so, j)
        #we can also update all the W of objective function here. which can used to speed up the calculation of gxi() and Ei
        #calcWs(so.alphas,so.X,so.Y)
        return 1
    else:
        return 0


# the outer loop of smo algrithm
def outterLoop(so,maxIter):
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:           #go over all
            for i in range(so.n):        
                alphaPairsChanged += takeStep(so,i)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:                   #go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((so.alphas.A > 0) * (so.alphas.A < so.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += takeStep(so,i)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return so.b,so.alphas



if __name__ == "__main__":
    X,Y = loadSample("/home/fandyst/Downloads/machinelearninginaction/Ch06/testSet.txt")
    print type(Y[0])
    so = smoObj(X,Y,100.,0.001)
    print outterLoop(so,1)
    x1 = [x1[0] for x1 in X]
    x2 = [x[1] for x in X]
    for i in xrange(len(x1)):
        if Y[i] < 0:
            scatter(x1[i],x2[i],color='green')
        else:
            scatter(x1[i],x2[i],color='red')
    #w = calcWs(so.alphas,so.X,so.Y)
    #x = np.linspace(-2,12,100)
    #y = w*x+so.b
    #print len(y)
    #scatter(x,y)
    #pl.show()

    www1 = calcWs(so.alphas,so.X,so.Y)
    w = float(-www1[0][0]/www1[1][0])
    b =  float(-(so.b / www1[1][0]))
    r = 1 / www1[1][0]
    lp_x1 = [-2., 12.]
    lp_x2 = []
    lp_x2up = []
    lp_x2down = []
    for x11 in lp_x1:
        lp_x2.append(w * x11 + b)
        lp_x2up.append(w * x11 + b + r)
        lp_x2down.append(w * x11 + b - r)

    plt.plot(lp_x1, lp_x2,
    plt.plot(lp_x1, lp_x2up, 'b--')
    plt.plot(lp_x1, lp_x2down, 'b--')








