# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from src.modules.convolution import LightConv1D
from src.modules.attention import SelfAttentionWithNormAndResidual
from src.modules.feed_forward import FeedForward
from src.modules.positionwise_feed_forward import TransformerFeedForward
import dataclasses
from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

JTensor = jnp.ndarray


class ConformerLayer(nn.Module):
    """Conformer layer as in https://arxiv.org/abs/2005.08100.

    Canonical version (with default params.)
      x = x + 1/2 * FFN(x)
      x = x + MHSA(x)
      x = x + Lconv(x)
      x = x + 1/2 * FFN(x)
      y = ln(x)

    Residual connections are implemented inside each individual block:
      FFN, MHSA, LConv.
    Optionally one can change the order of MHSA and conv.

    Attributes:
      model_dims: Encoder model dimension.
      kernel_size: Conv kernel size.
      ff_activation: Activation function used in the feedforward network.
      ff_residual_weight: Residual weight used in the fflayer.
      ffn_dim_multiplier: Feed forward hidden dimension will be ffn_dim_multiplier
        * model_dims.
      atten_num_heads: Number of attention heads.
      layer_order: Only mhsa, conv, mhsa_before_conv or conv_before_mhsa are
        supported
      dropout_prob: Dropout prob of inner components.
      conv_residual_dropout: Conv block residual dropout. Will be overwritten by
        p.dropout if it is not None.
      atten_residual_dropout: Attention block residual dropout. Will be
        overwritten by p.dropout if it is not None.
      ffn_residual_dropout: Feed forward block residual dropout. Will be
        overwritten by p.dropout if it is not None.
      atten_dropout: Dropout in Attention layer. Will be overwritten by p.dropout
        if it is not None.
      ffn_relu_dropout: Post activation dropout in Feed-forward layer. Will be
        overwritten by p.dropout if it is not None.
      fflayer_weight_sharing: If True, will ignore `fflayer_end_tpl`, and will
        make the fflayer_end layer as a weight-shared copy of the fflayer_start
        layer.
    """

    model_dims: int = 512
    kernel_size: int = 32
    ff_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    ff_residual_weight: float = 0.5
    ffn_dim_multiplier: int = 4
    atten_num_heads: int = 8
    layer_order: str = "mhsa_before_conv"
    dropout_prob: float | None = None
    conv_residual_dropout: float | None = None
    atten_residual_dropout: float | None = None
    ffn_residual_dropout: float | None = None
    atten_dropout: float | None = None
    ffn_relu_dropout: float | None = None
    fflayer_weight_sharing: bool = False
    downsample: bool = False
    skip_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        train: bool,
        use_running_average: bool | None = None,
        atten_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Conformer layer.

        Args:
          inputs: Input sequence jnp.ndarray of shape [B, T, H].
          train: Whether this is training. This affects Dropout behavior, and also
            affects BatchNorm behavior if 'use_running_average' is set to None.
          use_running_average: Optional, used to decide whether to use running
            statistics in BatchNorm (test mode), or the current batch's statistics
            (train mode). If not specified (or specified to None), default to 'not
            train'.
          atten_mask: Input jnp.ndarray attention mask.

        Raises:
          RuntimeError: if an attention mask is given but there's no attention layer

        Returns:
          The conformer output with shape [B, T, D].
        """
        if use_running_average is None:
            use_running_average = not train

        layer_order_set = ["mhsa", "conv",
                           "mhsa_before_conv", "conv_before_mhsa"]
        if self.layer_order not in layer_order_set:
            raise ValueError(
                f"`self.layer_order` must be within `{layer_order_set}`."
            )

        input_dims = inputs.shape[-1]

        # Set up the first ff layer.
        fflayer_start = TransformerFeedForward(
            name="fflayer_start",
            activation=self.ff_activation,
            input_dims=input_dims,
            hidden_dims=input_dims * self.ffn_dim_multiplier,
            residual_weight=self.ff_residual_weight,
            residual_dropout_prob=self.ffn_residual_dropout,
            relu_dropout_prob=self.ffn_relu_dropout,
        )

        # Set up the last ff layer.
        fflayer_end = TransformerFeedForward(
            name="fflayer_end",
            activation=self.ff_activation,
            input_dims=self.model_dims,
            hidden_dims=self.model_dims * self.ffn_dim_multiplier,
            residual_weight=self.ff_residual_weight,
            residual_dropout_prob=self.ffn_residual_dropout,
            relu_dropout_prob=self.ffn_relu_dropout,
        )

        # Setup attention layer.
        if "mhsa" in self.layer_order:
            trans_atten = SelfAttentionWithNormAndResidual(
                residual_dropout_prob=self.atten_residual_dropout,
                atten_dropout_prob=self.atten_dropout,
                num_heads=self.atten_num_heads,
            )

        # Setup convolution layer.
        lconv = LightConv1D(
            input_dims=self.model_dims,
            kernel_size=self.kernel_size,
            dropout_prob=self.conv_residual_dropout,
            downsample=self.downsample,
        )

        if not self.skip_layer_norm:
            final_ln = nn.LayerNorm(name="final_ln")

        if atten_mask is not None and "mhsa" not in self.layer_order:
            raise RuntimeError(
                "Attention mask is provided but no attention layer.")

        inputs = fflayer_start(inputs, train)

        if self.layer_order == "mhsa":
            inputs = trans_atten(
                inputs=inputs, train=train, atten_mask=atten_mask)
        elif self.layer_order == "conv":
            inputs = lconv(
                inputs, train=train, use_running_average=use_running_average
            )
        elif self.layer_order == "mhsa_before_conv":
            inputs = trans_atten(
                inputs=inputs, train=train, atten_mask=atten_mask)
            inputs = lconv(inputs, train)
        else:
            inputs = lconv(inputs, train)
            inputs = trans_atten(
                inputs=inputs, train=train, atten_mask=atten_mask)

        if self.fflayer_weight_sharing:
            # With the weight sharing, we apply fflayer_start again
            inputs = fflayer_start(inputs, train)
        else:
            inputs = fflayer_end(inputs, train)

        if not self.skip_layer_norm:
            inputs = final_ln(inputs)
        return inputs


class Conformer(nn.Module):
    """Projection layer followed by a conformer layer."""

    model_dims: int = 512
    kernel_size: int = 32
    ff_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    ff_residual_weight: float = 0.5
    ffn_dim_multiplier: int = 4
    atten_num_heads: int = 8
    layer_order: str = 'mhsa_before_conv'
    dropout_prob: float | None = None
    conv_residual_dropout: float | None = None
    atten_residual_dropout: float | None = None
    ffn_residual_dropout: float | None = None
    atten_dropout: float | None = None
    ffn_relu_dropout: float | None = None
    fflayer_weight_sharing: bool = False
    num_blocks: int = 1
    # tuples of layer index and corresponding scaling of number of channels
    downsample: list[tuple[int, float]] = dataclasses.field(
        default_factory=list)
    skip_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        train: bool,
        return_intermediate_list: bool,
        use_running_average: bool | None = None,
    ) -> jnp.ndarray:
        """Projection followed by a conformer layer.

        Args:
          inputs: Input sequence JTensor of shape [B, T, H].
          train: Whether this is training. This affects Dropout behavior, and also
            affects BatchNorm behavior if 'use_running_average' is set to None.
          return_intermediate_list: Whether to return a list of the activations
            after each conformer block, instead of only the final ones.
          use_running_average: Optional, used to decide whether to use running
            statistics in BatchNorm (test mode), or the current batch's statistics
            (train mode). If not specified (or specified to None), default to 'not
            train'.

        Returns:
          The conformer output with shape [B, T, D].
        """
        if use_running_average is None:
            use_running_average = not train
        if inputs.shape[-1] != self.model_dims:
            # Conformer requires the input dims to be `model_dims` so use a projection
            # layer that maps `input_dims` to `model_dims` before the conformer layer.
            inputs = FeedForward(output_dims=self.model_dims)(inputs)

        if self.dropout_prob is not None:
            all_dropouts = [
                self.atten_dropout,
                self.atten_residual_dropout,
                self.conv_residual_dropout,
                self.ffn_residual_dropout,
                self.ffn_relu_dropout,
            ]
            for prob in all_dropouts:
                assert prob is None or prob == self.dropout_prob

            atten_dropout = self.dropout_prob
            atten_residual_dropout = self.dropout_prob
            conv_residual_dropout = self.dropout_prob
            ffn_residual_dropout = self.dropout_prob
            ffn_relu_dropout = self.dropout_prob
        else:
            atten_dropout = self.atten_dropout
            atten_residual_dropout = self.atten_residual_dropout
            conv_residual_dropout = self.conv_residual_dropout
            ffn_residual_dropout = self.ffn_residual_dropout
            ffn_relu_dropout = self.ffn_relu_dropout

        intermediate = []
        model_dims = self.model_dims
        downsample = list(self.downsample).copy()
        for i in range(self.num_blocks):
            if downsample and downsample[0][0] == i:
                should_downsample = True
                model_dims = int(model_dims * self.downsample[0][1])
                model_dims = (model_dims // self.atten_num_heads) * \
                    self.atten_num_heads
                downsample = downsample[1:]
            else:
                should_downsample = False
            inputs = ConformerLayer(
                model_dims=model_dims,
                kernel_size=self.kernel_size,
                ff_activation=self.ff_activation,
                ff_residual_weight=self.ff_residual_weight,
                ffn_dim_multiplier=self.ffn_dim_multiplier,
                atten_num_heads=self.atten_num_heads,
                layer_order=self.layer_order,
                dropout_prob=self.dropout_prob,
                conv_residual_dropout=conv_residual_dropout,
                atten_residual_dropout=atten_residual_dropout,
                ffn_residual_dropout=ffn_residual_dropout,
                atten_dropout=atten_dropout,
                ffn_relu_dropout=ffn_relu_dropout,
                fflayer_weight_sharing=self.fflayer_weight_sharing,
                name='conformer_block_{}'.format(i),
                downsample=should_downsample,
                skip_layer_norm=self.skip_layer_norm,
            )(inputs, train, use_running_average=use_running_average)
            intermediate.append(inputs)
        if return_intermediate_list:
            return intermediate  # pytype: disable=bad-return-type  # jax-ndarray
        else:
            return inputs
