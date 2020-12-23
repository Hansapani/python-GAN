from keras.layers import Dense
from keras.models import Sequential


# define the standalone discriminator model
def define_discriminator(n_inputs: int=2) -> Sequential:
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model