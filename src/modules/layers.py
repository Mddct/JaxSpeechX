# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from src.modules.utils import ACT2FN
from typing import Callable

from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
import optax

from src.modules.feed_forward import FeedForward

JTensor = jnp.ndarray


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation layer.

    See "Squeeze-and-Excitation Networks" (Hu et al., 2018), particularly
    equations 2 and 3.

    Attributes:
      reduction_ratio: The reduction factor in the squeeze operation. Referred to
        as `r` in the paper.
      activation: The activation to apply after squeezing.
    """

    reduction_ratio: int = 4
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies SqueezeAndExcite on the 2D inputs.

        Args:
          inputs: Input data in shape of `(batch size, height, width, channels)`.

        Returns:
          JAX array with same shape as the input.
        """
        if inputs.ndim != 4:
            raise ValueError(
                "Inputs should in shape of `[batch size, height, width, features]`"
            )

        # Squeeze
        x = jnp.mean(inputs, axis=(1, 2))
        x = nn.Dense(features=x.shape[-1] //
                     self.reduction_ratio, name="Reduce")(x)
        x = self.activation(x)

        # Excite
        x = nn.Dense(features=inputs.shape[-1], name="Expand")(x)
        x = nn.sigmoid(x)
        return inputs * x[:, None, None, :]


class MBConv(nn.Module):
    """Mobile inverted bottleneck block.

    As introduced in "Mobilenetv2: Inverted residuals and linear bottlenecks"
    (Sandler et al., 2018). See figure 4d for an illustration and
    https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet for
    a reference implementation.

    The defaults are those from the MobileNetV2 paper. There is added support for
    batch normalization and squeeze-and-excitation blocks as used by EfficientNet.
    Note that the skip connection is not part of this module.

    Attributes:
      features: The number of filters.
      strides: The strides to use in the depthwise separable convolution.
      expand_ratio: The expansion factor to use. A block with expansion factor `N`
        is commonly referred to as MBConvN.
      kernel_size: The kernel size used by the depthwise separable convolution.
      activation: The activation function to use after the expanding 1x1
        convolution. Also used by the optional squeeze-and-excitation block.
      batch_norm: Whether to use batch normalization after the expanding and
        reducing convolutions.
      reduction_ratio: If given, a squeeze-and-excitation block is inserted after
        the depthwise separable convolution with the given reduction factor. Note
        that this reduction ratio is relative to the number of input channels,
        i.e., it scales with `expand_ratio`.
    """

    features: int
    strides: int
    expand_ratio: int
    kernel_size: tuple[int, int] = (3, 3)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu6
    batch_norm: bool = False
    reduction_ratio: int | None = None

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, use_running_average: bool = None
    ) -> jnp.ndarray:
        """Applies an inverted bottleneck block to the inputs.

        Args:
          inputs: Inputs should be of shape `(batch size, height, width, channels)`.
          use_running_average: Used to decide whether to use running statistics in
            BatchNorm (test mode), or the current batch's statistics (train mode).

        Returns:
          A JAX array of `(batch size, height, width, features)`.
        """
        features = self.expand_ratio * inputs.shape[-1]

        x = inputs
        if self.expand_ratio != 1:
            x = nn.Conv(
                features=features,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=False,
                name="ExpandConv",
            )(x)
            if self.batch_norm:
                x = nn.BatchNorm(
                    use_running_average=use_running_average, name="ExpandBatchNorm"
                )(x)
            x = self.activation(x)

        if self.strides == 2:

            def _pad_width(input_size: int, kernel_size: int) -> tuple[int, int]:
                """Calculate padding required to halve input with stride 2."""
                return (kernel_size // 2) - (1 - input_size % 2), kernel_size // 2

            padding = (
                _pad_width(x.shape[1], self.kernel_size[0]),
                _pad_width(x.shape[2], self.kernel_size[1]),
            )
        else:
            padding = "SAME"

        x = nn.Conv(
            features=features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=padding,
            feature_group_count=features,
            use_bias=False,
            name="DepthwiseConv",
        )(x)
        if self.batch_norm:
            x = nn.BatchNorm(
                use_running_average=use_running_average, name="DepthwiseBatchNorm"
            )(x)
        x = self.activation(x)

        if self.reduction_ratio is not None:
            x = SqueezeAndExcitation(
                reduction_ratio=self.reduction_ratio * self.expand_ratio,
                activation=self.activation,
            )(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=1,
            use_bias=False,
            name="ProjectConv",
        )(x)
        if self.batch_norm:
            x = nn.BatchNorm(
                use_running_average=use_running_average, name="ProjectBatchNorm"
            )(x)

        return x

class StridedAutopool(nn.Module):
    """Strided 1D Autopool over an array of shape [B, T, D].

    See https://arxiv.org/abs/1804.10070 for basic Autopool derivation.
    This implementation applies autopool to strided time windows.
    """

    alpha_0: float
    pool_width: int
    pool_stride: int
    padding: str

    @nn.compact
    def __call__(self, inputs):
        alpha_shape = [1] * (len(inputs.shape) - 1) + [inputs.shape[-1]]
        alpha = self.param(
            "alpha", nn.initializers.constant(self.alpha_0), alpha_shape
        )

        def pool_fn(x): return nn.pooling.avg_pool(  # pylint: disable=g-long-lambda
            x,
            window_shape=(self.pool_width,),
            strides=(self.pool_stride,),
            padding=self.padding,
        )
        exp_inputs = jnp.exp(alpha * inputs)
        auto_pooled = pool_fn(exp_inputs * inputs) / pool_fn(exp_inputs)
        return auto_pooled


class EarlyFeatureExtractor(nn.Module):
    """Network used as the "early feature extractor" for HuBERT.

    This module is comprised of a number of convolutional layers. It also uses
    group normalization after the first layer only. It is based on the
    architecture used for wav2vec 2.0 / HuBERT, and using the defaults of the
    implementation from
    https://github.com/facebookresearch/fairseq/blob/5307a0e078d7460003a86f4e2246d459d4706a1d/fairseq/models/wav2vec/wav2vec2.py

      Attributes:
        conv_layer_tuples: A List of (dim, kernel size, stride) tuples, one for
          each of the convolutional layers.
        dropout_prob: A float. The dropout probability.
        activation: The activation to apply after each convolutional "block".
        deprecated_group_conv: Whether to use the older version of this layer
          (which used grouped convolutions), for compatibility with old
          experiments. This option will be removed in the future.
    """

    conv_layer_tuples: tuple[tuple[int, int, int], ...]
    dropout_prob: float = 0.0
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    deprecated_group_conv: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Convolutional feature extractor used for "early" feature extraction.

        Args:
          inputs: Input sequence jnp.ndarray of shape [B, T, H].
          train: Whether we are in training mode. Affects dropout.

        Returns:
          A jnp.ndarray with shape [B, T, D].
        """
        if self.deprecated_group_conv:
            if inputs.ndim != 3:
                raise ValueError("Expected the input to have 3 dimensions.")
            model_dims = self.conv_layer_tuples[0][0]
            if inputs.shape[-1] != model_dims:
                inputs = FeedForward(output_dims=model_dims)(inputs)

        # TODO(etriantafillou): Experiment with adding residual connections.
        for i, (dim, k, stride) in enumerate(self.conv_layer_tuples):
            inputs = nn.Conv(
                features=dim,
                kernel_size=(k,),
                strides=(stride,),
                feature_group_count=dim if self.deprecated_group_conv else 1,
                use_bias=False,
                name="conv_layer_{}".format(i),
            )(inputs)

            inputs = nn.Dropout(self.dropout_prob)(
                inputs, deterministic=not train)

            if i == 0:
                if self.deprecated_group_conv:
                    inputs = nn.GroupNorm(
                        num_groups=None, group_size=dim)(inputs)
                else:
                    inputs = nn.GroupNorm(num_groups=dim)(inputs)

            inputs = self.activation(inputs)

        return inputs


def hinge_loss(predictor_outputs, targets):
    """Computes the hinge loss while accommodating targets in {0, 1}."""
    targets = 2 * targets - 1
    return optax.hinge_loss(predictor_outputs, targets)
