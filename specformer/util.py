from functools import partial

import haiku
import jax
import jax.numpy as jnp
import numpy as np


def apply_rotary_embedding(x):
    def fixed_pos_embedding(x):
        d = x.shape[-1]
        ifreq = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
        poses = np.arange(x.shape[-2])
        theta = np.einsum("i,j->ij", poses, ifreq)
        return np.sin(theta), np.cos(theta)

    def rotate_every_two(x):
        x1 = x[..., :, ::2]
        x2 = x[..., :, 1::2]
        x = jnp.stack((-x2, x1), axis=-1)
        return x.reshape(*x.shape[:-2], -1)

    x = x[..., : x.shape[-2] % -4, :]
    sin, cos = fixed_pos_embedding(x)
    sin, cos = map(partial(jnp.repeat, repeats=2, axis=-1), (sin, cos))
    return (x * cos) + (rotate_every_two(x) * sin)


def strip_pool(r, x):
    return sum(x[..., i::r] for i in range(r)) / r


def spectrogram(a, n_fft=128, window=jnp.hanning, eps=1e-12):
    n = n_fft
    rpad = -a.shape[-1] % n
    wins = jnp.pad(a, (0, rpad)).reshape(-1, n) * window(n)
    fftc = jnp.fft.fftshift(jnp.fft.fft(wins, n=n))[..., n // 2 : n]
    fftr = jnp.real(fftc * jnp.conj(fftc))
    return jnp.log(jnp.maximum(fftr, eps))


def ws_getter(next_getter, value, context):
    def standardize_weight(weight, eps=1e-4, prefix=""):
        mean = jnp.mean(weight, axis=-1, keepdims=True)
        var = jnp.var(weight, axis=-1, keepdims=True)
        gain = haiku.get_parameter(
            prefix + "_gain", shape=weight.shape[-1:], dtype=weight.dtype, init=jnp.ones
        )
        fanin = np.prod(weight.shape[:-1])
        scale = jax.lax.rsqrt(jnp.maximum(fanin * var, eps)) * gain
        shift = mean * scale
        return weight * scale - shift

    if not context.full_name.endswith("_gain"):
        value = standardize_weight(value, prefix=context.full_name)

    return next_getter(value)
