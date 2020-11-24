from numpy import hstack
from numpy import ones
from numpy import zeros
from numpy.random import rand 

def generate_real_samples(n):
    X1 = rand(n) - 0.5
    X2 = X1*X1

    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)

    X = hstack((X1, X2))

    y = ones((n,1))

    return X, y

def generate_fake_samples(n):
    X1 = -1 + rand(n) * 2
    X2 = -1 + rand(n) * 2

    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    X = hstack((X1, X2))

    y = zeros((n,1))

    return X, y