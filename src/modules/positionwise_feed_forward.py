# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

JTensor = jnp.ndarray

from src.modules.feed_forward import FeedForward
from src.modules.identity import Identity


# Transformer layers.
class TransformerFeedForward(nn.Module):
    """Transformer feedforward layer with residual connection and dropout.

    Attributes:
      input_dims: Depth of the input.
      hidden_dims: Hidden dimension of FFN.
      activation: Activation function to use. Options are RELU, RELU6, RELU^2,
        RELU^3, SIGMOID, TANH, GELU, GATED_GELU, GATED_SILU, NONE.
      residual_dropout_prob: Residual dropout.
      relu_dropout_prob: FFN dropout.
      add_skip_connection: Whether to add residual connection.
      residual_weight: Weight of the residual connection. Output = fn(x) *
        residual_weight + x.
    """

    input_dims: int = 0
    hidden_dims: int = 0
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    residual_dropout_prob: float = 0.0
    relu_dropout_prob: float = 0.0
    add_skip_connection: bool = True
    residual_weight: float = 1.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
        output_dims = self.input_dims
        inputs_normalized = nn.LayerNorm(name="layer_norm")(inputs)

        # Apply first FFN layer
        projected_inputs = FeedForward(
            output_dims=self.hidden_dims, activation=self.activation
        )(inputs_normalized)

        # Apply RELU dropout
        projected_inputs = nn.Dropout(self.relu_dropout_prob)(
            projected_inputs, deterministic=not train
        )

        # Apply second FFN layer
        projected_inputs = FeedForward(
            output_dims=output_dims, activation=Identity()
        )(projected_inputs)

        # Apply residual dropout
        projected_inputs = nn.Dropout(self.residual_dropout_prob)(
            projected_inputs, deterministic=not train
        )

        # Apply skip connection
        if self.add_skip_connection:
            projected_inputs = inputs + projected_inputs * self.residual_weight

        return projected_inputs