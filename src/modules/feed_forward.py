# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

JTensor = jnp.ndarray

class FeedForward(nn.Module):
    """Linear layer.

    Attributes:
      output_dims: Depth of the output.
      activation: The activation to apply after the linear layer.
    """

    output_dims: int = 0
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a feed forward layer to inputs.

        Args:
          inputs: The inputs jnp.ndarray.  Shaped [..., input_dims].

        Returns:
          Outputs. Shaped [..., output_dims].
        """
        x = nn.Dense(features=self.output_dims, name="FeedForward")(inputs)
        x = self.activation(x)
        return x