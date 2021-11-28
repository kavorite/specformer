# SpecFormer

This codebase is a transformer model for processing spectrograms. I implemented it because I was inspired by [Hourglass][hourglass] and [Audio Transformers][audio-xformers] to implement a model for an audio classification task. Incremental downsampling gives it more capacity to extract hierarchical, task-specific features in its penultimate layers. By introducing a broader receptive field and more nonlinearity where the model needs it most, we gain generalization performance advantages while still enjoying all the training and inference latency that come with lower input resolution, much like a ResNet.

```python
import haiku

from specformer import spectrogram, SpecFormer

@haiku.transform
def audio_projector(mono_samples):
    """
    project any real-valued signal to a fixed-width latent space by decomposing
    it into a human-friendly visualization of the frequency domain over time.
    """
    z = SpecFormer()(spectrogram(mono_samples).T)
    return haiku.Linear()(z.mean(axis=-1))

```

Additional notes for working with this repository:

- Unlike Hourglass, upsampling is not included: I have deemed it unnecessary. My intended application is supervised classification, so I'd just be downsampling output features again to plug them into a linear operator anyway, although upsampling back to the original signal resolution could be useful for training with unsupervised objectives, like the generative language games used for unsupervised training of masked language models in NLU.
- Unlike Audio Transformers, the model does not carry out its own feature extraction; instead I apply causal attention to a log-scale short-time Fourier transform of the target signal.
- Unlike both of the above-cited influences, **this model does not include any intermediate normalization.** It's designed to be fit using [superconvergent one-cycle policies][superconvergence] using weight standardization inspired by [NFNets] instead. See [optax] for the adaptive gradient clipping by unit-wise norm used to stably carry out optimization in NFNets.
- While transformers tend to be overparametrized, **this model does not include any intermediate dropout.** It is best for efficient generalization from small datasets, particularly in supervised tasks, if regularization is employed in these cases to prevent such architectures from memorizing specific instances of their inputs. For that reason the training script consuming this repository will also be employing [my implementation][mysam] of [sharpness-aware minimization][sam]. You've been warned.

[audio-xformers]: https://arxiv.org/abs/2105.00335
[hourglass]: https://arxiv.org/pdf/2110.13711v1.pdf
[superconvergence]: https://arxiv.org/abs/1708.07120
[nfnets]: https://arxiv.org/abs/2102.06171
[optax]: https://github.com/deepmind/optax
[sam]: https://arxiv.org/abs/2010.01412
[mysam]: https://github.com/kavorite/sam
