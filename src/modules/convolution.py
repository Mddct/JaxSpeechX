# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from typing import Callable

from flax import linen as nn
import jax
from jax import numpy as jnp

JTensor = jnp.ndarray

from src.modules.feed_forward import FeedForward
from src.modules.identity import Identity

# Convolution layers.
class LightConv1D(nn.Module):
    """Lightweight conv layer.

    architecture::

    input-ln()-ff()-glu()-depthwise_conv1d()-norm()-act()-ff()-dropout()-+-output
      |__________________________________________________________________|

    Attributes:
      input_dims:      Input and (in fact,) output dimension.
      kernel_size:     Kernel size of 1d deptwise conv.
      conv_activation: Activation after normalization.
      dropout_prob:    Dropout probability.
    """

    input_dims: int | None = None
    kernel_size: int | None = None
    conv_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    dropout_prob: float = 0.0
    downsample: bool = True

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        train: bool,
        use_running_average: bool | None = None,
    ) -> jnp.ndarray:
        """Lightweight conv layer.

        Args:
          inputs: Input sequence jnp.ndarray of shape [B, T, H].
          train: Whether this is training. This affects Dropout behavior, and also
            affects BatchNorm behavior if 'use_running_average' is set to None.
          use_running_average: Optional, used to decide whether to use running
            statistics in BatchNorm (test mode), or the current batch's statistics
            (train mode). If not specified (or specified to None), default to 'not
            train'.

        Returns:
          The lconv output with shape [B, T, H].
        """
        if use_running_average is None:
            use_running_average = not train
        unnormalized_inputs = inputs

        inputs = nn.LayerNorm(name="ln")(inputs)
        act_inputs = FeedForward(
            output_dims=self.input_dims, activation=Identity()
        )(inputs)
        gated_inputs = FeedForward(
            output_dims=self.input_dims, activation=Identity()
        )(inputs)
        inputs = act_inputs * jax.nn.sigmoid(gated_inputs)

        inputs = nn.Conv(
            features=self.input_dims,
            kernel_size=(self.kernel_size,),
            strides=2 if self.downsample else 1,
            padding="SAME",
            input_dilation=1,
            kernel_dilation=1,
            feature_group_count=self.input_dims,
            use_bias=False,
        )(inputs)

        inputs = nn.BatchNorm()(inputs, use_running_average=use_running_average)
        inputs = self.conv_activation(inputs)

        inputs = FeedForward(output_dims=self.input_dims, activation=Identity())(
            inputs
        )
        inputs = nn.Dropout(self.dropout_prob)(inputs, deterministic=not train)

        if self.downsample:
            unnormalized_inputs = nn.avg_pool(
                unnormalized_inputs, (2,), (2,), padding="SAME"
            )
            # If downsampling happened, the dimensions might also have changed, which
            # means we need to project the inputs for the residual connection
            if unnormalized_inputs.shape[-1] != self.input_dims:
                unnormalized_inputs = nn.Dense(features=self.input_dims)(
                    unnormalized_inputs
                )

        output = inputs + unnormalized_inputs
        return output
