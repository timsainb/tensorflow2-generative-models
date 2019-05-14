[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/timsainb/tensorflow2-generative-models/master)

Generative models in Tensorflow 2
==============================

[Tim Sainburg](https://timsainburg.com/) (PhD Candidate, UCSD, Gentner Laboratory)

This is a small project to implement a number of generative models in Tensorflow 2. Layers and optimizers use Keras. The models are implemented for two datasets: [fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), and [NSYNTH](https://magenta.tensorflow.org/datasets/nsynth). Networks were written with the goal of being as simple and consistent as possible while still being readable. Because each network is self contained within the notebook, they should be easily run in a colab session. 

## Included models:
### Autoencoder (AE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/0.0-Autoencoder-fashion-mnist.ipynb)

A simple autoencoder network.

![an autoencoder](imgs/ae.png)

### Variational Autoencoder (VAE) ([article](https://arxiv.org/abs/1312.6114)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/1.0-Variational-Autoencoder-fashion-mnist.ipynb)

The original variational autoencoder network,  using [tensorflow_probability](https://github.com/tensorflow/probability)

![variational autoencoder](imgs/vae.png)

### Generative Adversarial Network (GAN) ([article](https://arxiv.org/abs/1406.2661)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/2.0-GAN-fashion-mnist.ipynb)

GANs are a form of neural network in which two sub-networks (the encoder and decoder) are trained on opposing loss functions: an encoder that is trained to produce data which is indiscernable from the true data, and a decoder that is trained to discriminate between the data and generated data.

![gan](imgs/gan.png)

### Wasserstein GAN with Gradient Penalty (WGAN-GP) ([article](https://arxiv.org/abs/1701.07875)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb)

WGAN-GP is a GAN that improves over the original loss function to improve training stability. 

![wgan gp](imgs/gan.png)


### VAE-GAN ([article](https://arxiv.org/abs/1512.09300)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/6.0-VAE-GAN-fashion-mnist.ipynb)

VAE-GAN combines the VAE and GAN to autoencode over a latent representation of data in the generator to improve over the pixelwise error function used in autoencoders. 

![vae gan](imgs/vaegan.png)

### Generative adversarial interpolative autoencoder (GAIA) ([article](https://arxiv.org/abs/1807.06650)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/5.0-GAIA-fashion-mnist.ipynb)

GAIA is an autoencoder trained to learn convex latent representations by adversarially training on interpolations in latent space projections of real data. 

![generative adversarial interpolative autoencoding network](imgs/gaia.png)

## Other Notebooks:

### Seq2Seq Autoencoder (without attention) (Fasion MNIST: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/4.0-seq2seq-fashion-mnist.ipynb) | NSYNTH: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/9.0-seq2seq-NSYNTH.ipynb))

Seq2Seq models use recurrent neural network cells (like LSTMs) to better capture sequential organization in data. This implementation uses Convolutional Layers as input to the LSTM cells, and a single Bidirectional LSTM layer. 

![a seq2seq bidirectional lstm in tensorflow 2.0](imgs/seq2seq.png)

### Spectrogramming, Mel Scaling, MFCCs, and Inversion in Tensorflow [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/7.0-Tensorflow-spectrograms-and-inversion.ipynb)

Tensorflow has a signal processing package that allows us to generate spectrograms from waveforms as part of our dataset iterator, rather than pregenerating a second spectrogram dataset. This notebook can serve as a reference for how this is done. Spectrogram inversion is done using the Griffin-Lim algorithm. 

![spectrogram inversion in tensorflow 2.0](imgs/spectrogram-inversion.png)


### Iterator for NSynth [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/8.0-NSYNTH-iterator.ipynb)

The NSYNTH dataset is a set of thousands of musical notes saved as waveforms. To input these into a Seq2Seq model as spectrograms, I wrote a small dataset class that converts to spectrogram in tensorflow (using the code from the spectrogramming notebook). 

![a dataset iterator for tensorflow 2.0](imgs/nsynth-dataset.png)
