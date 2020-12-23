from matplotlib import pyplot
from numpy import ones
from keras.models import Sequential

import os
import DiscriminatorModel as dm
import GeneratorModel as gm
import SampleGenerator as sg


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator: Sequential, discriminator: Sequential) -> Sequential:
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch: int, generator: Sequential, discriminator: Sequential, latent_dim: int, n: int=100):
    # prepare real samples
    x_real, y_real = sg.generate_real_samples(n)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = sg.generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    # scatter plot real and fake data points
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')

    #create director if it doens't exist
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    # save plot to file
    filename = './imgs/generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


# train the generator and discriminator
def train(g_model: Sequential, d_model: Sequential, gan_model: Sequential, latent_dim: int, n_epochs: int=10000, n_batch: int=128, n_eval: int=2000):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = sg.generate_real_samples(half_batch)
        # prepare fake examples
        x_fake, y_fake = sg.generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = sg.generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator✬s error
        gan_model.train_on_batch(x_gan, y_gan)
        # evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim)


# size of the latent space
latent_dim = 5

# create the discriminator
discriminator = dm.define_discriminator()

# create the generator
generator = gm.define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)

# train model
train(generator, discriminator, gan_model, latent_dim)