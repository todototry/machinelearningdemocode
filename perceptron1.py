#encoding:utf-8
# features
x = array([[ 4,  1],
       [ 1,  6],
       [13,  1],
       [15, 17],
       [ 5,  9],
       [15, 10],
       [ 6, 16],
       [ 7, 10],
       [ 8, 16],
       [13,  1],
       [13,  3],
       [10, 19],
       [16,  2],
       [19,  5],
       [17,  9],
       [ 8, 17],
       [ 6, 13],
       [ 5, 13],
       [16,  6],
       [18, 17],
       [ 7,  5],
       [14,  3],
       [ 2,  6],
       [17,  6],
       [17, 10],
       [15,  8],
       [16, 16],
       [13, 11],
       [18,  9],
       [ 1,  2],
       [ 8,  2],
       [ 5, 15],
       [ 4, 13],
       [12,  8],
       [11, 13],
       [ 6,  1],
       [ 8,  5],
       [ 1, 15],
       [ 2,  9],
       [19,  5]])
#label
y = array([-1, -1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,
       -1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,
        1, -1, -1, -1, -1,  1])

#samples
sample = matrix([[ 4,  1, -1],
        [ 1,  6, -1],
        [13,  1,  1],
        [15, 17,  1],
        [ 5,  9, -1],
        [15, 10,  1],
        [ 6, 16,  1],
        [ 7, 10, -1],
        [ 8, 16,  1],
        [13,  1,  1],
        [13,  3,  1],
        [10, 19,  1],
        [16,  2,  1],
        [19,  5,  1],
        [17,  9,  1],
        [ 8, 17,  1],
        [ 6, 13, -1],
        [ 5, 13, -1],
        [16,  6,  1],
        [18, 17,  1],
        [ 7,  5, -1],
        [14,  3,  1],
        [ 2,  6, -1],
        [17,  6,  1],
        [17, 10,  1],
        [15,  8,  1],
        [16, 16,  1],
        [13, 11,  1],
        [18,  9,  1],
        [ 1,  2, -1],
        [ 8,  2, -1],
        [ 5, 15, -1],
        [ 4, 13, -1],
        [12,  8,  1],
        [11, 13,  1],
        [ 6,  1, -1],
        [ 8,  5, -1],
        [ 1, 15, -1],
        [ 2,  9, -1],
        [19,  5,  1]])

# 生成随机数据和快速标定方法  the method of generating random data and labeled class for perceptron. 
#xxx = random.randint(0,20,40)
#yyy = random.randint(0,20,40)
#label = [1 if (-2*xxx[i]+25)>0 else -1 for i in range(40)]
#label = asarray(label)
#scatter(xxx[label[:]==-1],yyy[label[:]==-1],c="green")
#scatter(xxx[label[:]==1],yyy[label[:]==1],c="red")


#plot the data based on its label
scatter(x[y[:]==-1,0],x[y[:]==-1,1],c="green")
scatter(x[y[:]==1,0],x[y[:]==1,1],c="red")

def Loss(x,y,w,b):
    return y*(w[0]*x[0]+w[1]*x[1] + b)

def adjustPara(w,b,x,y):
    w[0]= w[0] + y*x[0]
    w[1]= w[1] + y*x[1]
    b = b + y
    return w,b

w = [0,0]
b = 0
error = -1
while error < 0:
    for index,(value1,value2) in enumerate(x):
        if Loss((value1,value2),y[index],w,b) <= 0:
            w,b = adjustPara(w,b,(value1,value2),y[index])
            error += 1
    if error < 0:
        break
    else:
        error = -1
w,b

#output: [10,5],-134
#model:  0 = 10x+5y-134

scatter(x[y[:]==-1,0],x[y[:]==-1,1],c="green")
scatter(x[y[:]==1,0],x[y[:]==1,1],c="red")
linex = linspace(0,25,200)
liney = [(i*w[0]+b)/-w[1] for i in linex]
plot(linex,liney)





