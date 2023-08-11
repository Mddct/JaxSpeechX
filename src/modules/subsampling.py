# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from flax import linen as nn
from jax import numpy as jnp

import math
import numpy as np
JTensor = jnp.ndarray


class PositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence.

    Attributes:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
    """

    embedding_dims: int = 0
    min_timescale: int = 1
    max_timescale: int = 10_000

    @nn.compact
    def __call__(self, seq_length: int) -> jnp.ndarray:
        """Generates an array of sinusoids with different frequencies.

        Args:
          seq_length: Sequence length of the embeddings to be generated.

        Returns:
          An array of shape (1, seq_length, embedding_dim) containing positional
          embeddings.
        """
        position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(
            self.max_timescale / self.min_timescale
        ) / jnp.maximum(num_timescales - 1, 1.0)
        inv_timescales = self.min_timescale * jnp.exp(
            jnp.arange(num_timescales) * -log_timescale_increment
        )
        scaled_time = (
            position[:, :, jnp.newaxis]
            * inv_timescales[jnp.newaxis, jnp.newaxis, :]
        )
        signal = jnp.concatenate(
            [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2
        )
        # Force usage of `np` rather than `jnp` to compute static values at trace
        # time.
        if self.embedding_dims != 0:
            signal = jnp.pad(
                signal, [(0, 0), (0, 0), (0, np.mod(self.embedding_dims, 2))]
            )
        return signal


class ConvolutionalSubsampling(nn.Module):
    """Convolutional subsampling module.

    This is the convolutional subsampling module as used in the conformer
    paper[^1]. It consists of two 2D convolutional layers with a stride of 2.
    The frequencies and output channels get combined to produce a 1D output.
    Relative positional embeddings are added for the conformer blocks.

    [1]: Gulati, Anmol, et al. "Conformer: Convolution-augmented transformer for
      speech recognition." arXiv preprint arXiv:2005.08100 (2020).
    """

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (2, 2)
    num_layers: int = 2
    dropout_prob: float = 0.1

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Apply convolutional subsampling.

        Args:
          inputs: A batch of spectrograms of size (batch, time, channels).
          train: Whether or not this is training (used for dropout).

        Returns:
          A subsampled array that is 4 times small in the time and channels dims.
        """
        x = inputs

        # Subsample
        x = nn.Conv(self.features, self.kernel_size, strides=self.strides)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, self.kernel_size, strides=self.strides)(x)
        x = nn.relu(x)

        # Merge channels and frequency dimension
        x = jnp.reshape(x, x.shape[:-2] + (-1,))
        x = nn.Dense(self.features)(x)

        # Add positional embeddings
        x = x + PositionalEmbedding(embedding_dims=self.features)(x.shape[-2])
        x = nn.Dropout(self.dropout_prob, deterministic=not train)(x)

        return x
