from keras.layers import Dense
from keras.models import Sequential

import SampleGenerator as gen

#import tensorflow as tf
# import csv

# with open('train.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         print(', '.join(row))

# csvfile = open('train.csv', newline='')

# print(csvfile.readline())

def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train_discriminator(model, n_epochs=1000, n_batch=128):
    half_batch = int(n_batch / 2)

    for i in range (n_epochs):
        X_real, y_real = gen.generate_real_samples(half_batch)
        model.train_on_batch(X_real, y_real)

        X_fake, y_fake = gen.generate_fake_samples(half_batch)
        model.train_on_batch(X_fake, y_fake)

        _, acc_real = model.evaluate(X_real, y_real, verbose = 0)
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose = 0)

        print(i, acc_real, acc_fake)


model = define_discriminator()
train_discriminator(model)
