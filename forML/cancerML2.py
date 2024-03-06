from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target, stratify = cancer.target, test_size = 0.2, random_state = 42)

class LogisticNeuron:
    
    def forpass(self, x):
        z1 = np.dot(x * self.w1) + self.b1
        self.a1 = self.activation(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        return z2
    
    def backprop(self, x, err):
        m = len(x)
        w2_grad = np.dot(self.a1.T, err)/m
        b2_grad = np.sum(err)/m
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1-self.a1)
        w1_grad = np.dot(x.T, err_to_hidden)/m
        b1_grad = np.sum(err_to_hidden, axis=0)/m
        return w1_grad, b1_grad, w2_grad, b2_grad
    
    def init_weights(self, n_features):
        self.w1 = np.ones((n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.ones((self.units,1))
        self.b2 = 0

    def activation(self, z):
        a = 1/(1 + np.exp(-z))
        return a

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            loss = 0  #as we start one ite we store loss
            indexes = np.random.permutation(np.arrange(len(x)))
            for i in indexes:
                z =self.forpass(x_i)
                a = self.activation(z)
                err = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad
                a = np.clip(a, 1e-10,1-1e-10) #clip the loss before ccumlating it for safe log calculation
                loss += -(y[i]*np.log(a) + (1-y[i]) * np.log(1-a))
            self.losses.append(loss/len(y))
    def predict(self,x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5           

    def score(self, x, y):
        return np.mean(self.predict(x) == y)

    def __init__(self):
        self.w = None;
        self.b = None;
        self.loss = []


class RandomInitNetwrok(DualLayer):
    def init_weights(self, n_features):
        np.random.seed(42)
        self.w1 = np.random.normal(0,1,(n_features, self.units))
        self.w2 = np.random.normal(0,1,(self.units,1))
        self.b2 = 0

# I think it starts here
class MinibatchNetwork(RandomInitNetwrok):
    def __init__(self, units=10, batch_size=32, learning_rate = 0.1, l1=0, l2=0):
        super().__init__(units, learning_rate, l1, l2)
        self.batch_size = batch_size #3.11.15


singleLaywitLoss = LogisticNeuron()
singleLaywitLoss.fit(x_train, y_train)
print(singleLaywitLoss.score(x_test, y_test))
#accuracy = np.mean(runReg.predict(x_test) == y_test)
#print(f"Accu: {np.int16(accuracy * 100)}%")

    



