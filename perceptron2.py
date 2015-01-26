
#--------------------方法2-------------------

# -*- coding:gbk -*- 
 
#针对二维空间的感知机学习算法 
#感知机模型 f(x) = sign(w*x+b) 
#学习策略 损失函数: L(w,b) = -y(w*x+b) 
#学习算法 随即梯度下降 
# @author heshaopeng 
# @date 2013-01-15 
 
#定义感知机类 
class Perceptron: 
    #初始化类 
    #m_learnrate:学习率  m_w0:x0的权值  m_w1:x1的权值 m_b:常向量 
    def __init__(self,m_learnrate,m_w0,m_w1,m_b): 
        self.m_learnrate = m_learnrate 
        self.m_w0 = m_w0 
        self.m_w1 = m_w1 
        self.m_b = m_b 
 
    #判断针对训练数据x，估测的模型与实际数据是否有误差 
    #即判断损失函数L(w,b)是否为0 
    def judgeHasError(self,x): 
        #如果表达式小于0，说明没有被正确分类 
        #即y*(w0*x0+w1*x1+b)<0 
        result = x[2]*(x[0]*self.m_w0+x[1]*self.m_w1+self.m_b) 
        if result<=0: 
            return False 
        else: 
            return True 
 
    #有误差的话，调整模型参数 
    def adjustParam(self,x): 
        #根据梯度下降法调整参数 
        self.m_w0 = self.m_w0 + self.m_learnrate*x[2]*x[0] 
        self.m_w1 = self.m_w1 + self.m_learnrate*x[2]*x[1] 
        self.m_b = self.m_b + self.m_learnrate*x[2] 
        return 
 
    #训练数据集 
    def trainData(self,data,num): 
        count = 0 
        isOver = False 
        while not isOver: 
            print "w0  w1  b :" +str(self.m_w0)+" "+str(self.m_w1)+" "+str(self.m_b) 
            for i in range(0,num): 
                if not self.judgeHasError(data[i]): 
                    count = count+1 
                    print "调整次数："+ str(count) 
                    self.adjustParam(data[i]) 
                    isOver = False 
                    break 
                else: 
                    isOver = True 
        #打印最后的参数 
        print "w0  w1  b :" +str(self.m_w0)+" "+str(self.m_w1)+" "+str(self.m_b) 
 
if __name__ == '__main__': 
    p = Perceptron(1,0,0,0) 
    #data = [[3,3,1],[4,3,1],[1,1,-1]] 
    data = [[3,3,1],[4,3,1],[1,1,-1],[2,2,-1],[5,4,1],[1,3,-1]] 
    p.trainData(data,6) 

