import flax
import jax
import flax.linen as nn
from functools import partial

def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}