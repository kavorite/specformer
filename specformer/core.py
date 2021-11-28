from functools import partial

import haiku
import jax.numpy as jnp
import numpy as np

from .util import apply_rotary_embedding, strip_pool, ws_getter


class DecoderBlock(haiku.Module):
    def __init__(self, heads=2, out_d=None, key_d=None):
        super().__init__()
        self.h = heads
        self.d = out_d
        self.k = key_d

    @haiku.transparent
    def attend(self, x):
        if self.d is None:
            self.d = x.shape[-1]
        n = x.shape[-2]
        causal_mask = np.tril(np.ones((n, n)))
        d, h = self.d, self.h
        x = apply_rotary_embedding(x)
        q, k, v = (haiku.Linear(self.k or d)(x) for _ in range(3))
        attended = haiku.MultiHeadAttention(
            num_heads=h, key_size=d, w_init_scale=d ** -0.5
        )(q, k, v, causal_mask)
        return jnp.einsum("...nd,...nz->...nz", x, attended / h)

    def __call__(self, x):
        with haiku.custom_getter(ws_getter):
            x = jnp.einsum("...nd,...nz->...nz", x, self.attend(x))
            x = haiku.Linear(self.d)(x)
        return x


class SpecFormer(haiku.Module):
    def __init__(self, widths=[2, 4, 8], depths=[3, 3, 3], reductions=[2, 2, 2]):
        super().__init__()
        assert (
            len(set(map(len, (widths, depths, reductions)))) == 1
        ), "width, depth, and reduction specifiers must have the same stage count"
        self.blocks = []
        for width, depth, reduction in zip(widths, depths, reductions):
            for _ in range(depth):
                self.blocks.append(DecoderBlock(width))
            self.blocks.append(partial(strip_pool, r=reduction))

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x
