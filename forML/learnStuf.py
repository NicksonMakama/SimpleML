#from sklearn.linear_model import Perceptron
#X = [
#    [0,0],
#    [0,1],
#    [1,0],
#    [1,1]
#]
#y = [-1,1,1,1]
#
#p = Perceptron()
#p.fit(X,y)
#print("weight ", p.coef_, p.intercept_)
#
#W = [[0,1],[1,1]]
#
#print("New predict", p.predict(W))
#print("Accuracy ", p.score(X,y)*100)

# work with MNIST using MLP
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
import numpy as np

#Read, split MNIST into Traing & Test
mnist_ds = fetch_openml('mnist_784')
mnist_ds.data = mnist_ds.data/255.0 #scale the values to be frm (0,1) to ease learning
x_train = mnist_ds.data[:50000] # from : to
x_test = mnist_ds.data[50000:]
y_train = np.int16(mnist_ds.target[:50000])
y_test = np.int16(mnist_ds.target[50000:])

#training
mlpc = MLPClassifier(hidden_layer_sizes=(120), learning_rate_init=0.1, batch_size=512, max_iter=250, solver='adam', verbose=True) 
# jus 1 layer wit 1200 neurons eg (120, 60) 2 layer 120 nuron & 60 nuron , max-iter is no of Iteratn, batchSize is hw many data to use in each iter, solver is the optimzt algo, 
#verbose true to tell  us info about traing
mlpc.fit(x_train, y_train)

#predict with test data
res = mlpc.predict(x_test)   # it'll give all the test data classies

#Get confusn Matrx so we can cal Accuracy
conFM =  np.zeros((10,10), dtype=np.int16)
for i in range(len(res)):
    conFM[res[i]][y_test[i]] +=1
print(conFM)

#Accuracy
no_correct = 0
for i in range(10):
    no_correct+=conFM[i][i]
accuracy= no_correct/len(res)
print("Accuracy ", accuracy*100,"%")








