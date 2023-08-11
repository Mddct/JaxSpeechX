# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from flax import linen as nn
from jax import numpy as jnp

JTensor = jnp.ndarray

class Identity(nn.Module):
    """Identity layer."""

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """Identity function.

        Args:
          inputs: Input array.
          *args: Any other arguments are ignored.
          **kwargs: Any keyword arguments are ignored.

        Returns:
          The input, unchanged.
        """
        return inputs