import numpy as np

#data&model
env = 0.3
n0 = 2
n1 = 8
n2 = 3
m = 10
A0 = np.random.rand(n0, m)
Y = np.random.rand(n2, m)

#parameters
w1 = np.random.rand(n1, n0) * 0.1
b1 = np.random.rand(n1, 1)
w2 = np.random.rand(n2, n1) * 0.1
b2 = np.random.rand(n2, 1)

for i in range(100000):

    #forward propagation
    Z1 = np.dot(w1, A0) + b1
    A1 = 1/(1+np.exp(-1*Z1))
    Z2 = np.dot(w2, A1) + b2
    A2 = 1/(1+np.exp(-1*Z2))
    c = (1/m) * np.sum(
        -1*(Y*np.log10(A2)+(1-Y)*np.log10(1-A2)),
        axis=1,
        keepdims=True
    )

    #backpropagation
    dZ2 = A2 - Y
    dw2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(w2.T, dZ2) * A1 * (1 - A1)
    dw1 = (1/m) * np.dot(dZ1, A0.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    b1 -= env * db1
    w1 -= env * dw1
    b2 -= env * db2
    w2 -= env * dw2

print("\nc " + str(c.shape) + " = " + str(c))
print("\nY " + str(Y.shape) + " = " + str(Y))
print("\nYhat " + str(A2.shape) + " = " + str(A2))
print("\nb1 " + str(b1.shape) + " = " + str(b1))
print("\nb2 " + str(b2.shape) + " = " + str(b2))
print("\nw1 " + str(w1.shape) + " = " + str(w1))
print("\nw2 " + str(w2.shape) + " = " + str(w2))
print("\n")