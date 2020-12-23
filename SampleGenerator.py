from keras.models import Sequential
from numpy import array
from numpy import hstack
from numpy import ones
from numpy import zeros
from numpy.random import rand 
from numpy.random import randn
from typing import Tuple


# generate n real samples with class labels
def generate_real_samples(n: int) -> Tuple[array, array]:
    # generate inputs in [-0.5, 0.5]
    X1 = rand(n) - 0.5
    # generate outputs X^2
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = ones((n, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim: int, n: int) -> array:
    # generate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator: Sequential, latent_dim: int, n: int) -> Tuple[array, array]:
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n, 1))
    return X, y