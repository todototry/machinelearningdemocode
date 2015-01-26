# coding: utf-8
def getSamples(filename):
    f = open(filename)
    filelen = len(f.readlines())
    sampleMat = zeros((filelen,3))
    sampleLabel = []
    for index,line in enumerate(f.readlines()):
        line = line.strip()
        sampleData = line.split('\t')
        sampleMat[index,:] = sampleData[0:3]
        sampleLabel.append(sample[-1])
        index += 1
    return sampleMat,sampleLabel