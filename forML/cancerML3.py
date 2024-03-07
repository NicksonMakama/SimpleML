from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class DualLayer:   # This is a simple Neural Network 1 layer that has two perceptrons in it
    
    def init_weights(self, n_features): #lets declare the things we will use
        self.w1 = np.ones((n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.ones((self.units, 1))
        self.b2 = 0

    def forpass(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.activation(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2 #notice how to connect two neurons by puting output of the first into the second
        return z2

    def backprop(self, x, err):
        m = len(x)
        w2_grad = np.dot(self.a1.T, err) / m
        b2_grad = np.sum(err) / m
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
        w1_grad = np.dot(x.T, err_to_hidden) / m
        b1_grad = np.sum(err_to_hidden, axis=0) / m
        return w1_grad, b1_grad, w2_grad, b2_grad


    def activation(self, z):
        a = 1 / (1 + np.exp(-z)) #A sigmoid Activation func in Layer 1
        return a

    def fit(self, x, y, epochs=100): # main training that will learn the w and b in Layer 1
        self.init_weights(x.shape[1])
        for epoch in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                x_i, y_i = x[i], y[i]
                z = self.forpass(x_i)
                a = self.activation(z)
                err = -(y_i - a)
                w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x_i, err)
                self.w1 -= w1_grad
                self.b1 -= b1_grad
                self.w2 -= w2_grad
                self.b2 -= b2_grad
                a = np.clip(a, 1e-10, 1-1e-10)  # Clip the loss for safe log calculation
                loss += -(y_i * np.log(a) + (1 - y_i) * np.log(1 - a))
            self.losses.append(loss / len(y))

    def predict(self, x):  #it tries to predict a new case and store it as z
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z)) #we apply same activation for the new case
        return a > 0.5

    def score(self, x, y):
        return np.mean(self.predict(x) == y) 

    def __init__(self, units=10):
        self.units = units
        self.losses = []
        self.init_weights(units)

class RandomInitNetwork(DualLayer):  #this inheritance means RandInitNet is a DualLayer class so it has all the methods & vars
    def init_weights(self, n_features):
        np.random.seed(42) #seed is like the first no the generator gets to generate other numbers from
        self.w1 = np.random.normal(0, 1, (n_features, self.units))
        self.w2 = np.random.normal(0, 1, (self.units, 1))
        self.b2 = 0

class MinibatchNetwork(RandomInitNetwork):  # inheritance means this MiniBatNet is a RandomInitNet
    def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):  #l1 & l2 are regularlizatn parameters to add to w1 & w2 to avoid overfitting
        super().__init__(units)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.b1 = np.zeros(self.units)  # Add this line to initialize b1

    def training(self, x, y, m):
        z = self.forpass(x)
        a = self.activation(z)
        err = -(y - a)
        w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
        # Update weights with regularization terms
        w1_grad -= (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m
        w2_grad -= self.l2 * self.w2 / m
        self.w1 -= self.learning_rate * w1_grad
        self.b1 -= self.learning_rate * b1_grad  # Corrected attribute name to b1
        self.w2 -= self.learning_rate * w2_grad
        self.b2 -= self.learning_rate * b2_grad
        return a

    
    def fit(self, x, y, epochs=100, x_val=None, y_val=None): # try to predict
        y = y.reshape(-1, 1)
        self.init_weights(x.shape[1])
        np.random.seed(42)
        for epoch in range(epochs):
            loss = 0
            for x_batch, y_batch in self.gen_batch(x, y):
                y_batch = y_batch.reshape(-1, 1)
                m = len(x_batch)
                a = self.training(x_batch, y_batch, m)

    def gen_batch(self, x, y):
        length = len(x)
        bins = length // self.batch_size
        if length % self.batch_size:
            bins += 1
        indexes = np.random.permutation(np.arange(len(x)))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]



# Load breast cancer dataset
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, test_size=0.2, random_state=42)

# Assume you have preprocessed your data (scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

minibatch_net = MinibatchNetwork(batch_size=128, l2=0.01)
minibatch_net.fit(x_train_scaled, y_train, epochs=500)
print(minibatch_net.score(x_test_scaled, y_test))
