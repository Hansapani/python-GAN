from keras.models import Sequential
from keras.layers import Dense


# define the standalone generator model
def define_generator(latent_dim: int, n_outputs: int=2) -> Sequential:
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model