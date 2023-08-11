# https://github.com/google-research/chirp/blob/main/chirp/models/conformer.py

from flax import linen as nn
from jax import numpy as jnp

JTensor = jnp.ndarray

# Self-attention layers.
class SelfAttentionWithNormAndResidual(nn.Module):
    """Self attention sub-layer used in the Conformer layer.

    Input is first normalized using layer norm. Output is processed using
    multi-headed attention. And finally, the output of the attention layer
    is combined with the input by residual connection.

    For the normalization, we can specify pre norm or post norm.
    For the residual connection, we can specify the residual weight.

    Attributes:
      residual_weight: Weight of the residual connection. Output = fn(x) *
        residual_weight + x * input_weight.
      input_weight: Weight of the input connection. Output = fn(x) *
        residual_weight + x * input_weight.
      pre_layer_norm: Whether to apply norm before or after the layer.
      residual_dropout_prob: Probability at which we apply dropout to the residual
        layers, such that, residual(x, y) = (x + dropout(y)).
    """

    residual_weight: float = 1.0
    input_weight: float = 1.0
    pre_layer_norm: bool = True
    residual_dropout_prob: float = 0.0
    atten_dropout_prob: float = 0.0
    num_heads: int = 1

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        train: bool,
        atten_mask: JTensor | None = None,
    ) -> jnp.ndarray:
        unnormalized_inputs = inputs

        if self.pre_layer_norm:
            inputs = nn.LayerNorm()(inputs)

        self_atten = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, dropout_rate=self.atten_dropout_prob
        )
        result = self_atten(
            inputs_q=inputs,
            inputs_kv=inputs,
            mask=atten_mask,
            deterministic=not train,
        )

        if not self.pre_layer_norm:
            result = nn.LayerNorm()(result)

        dropout = nn.Dropout(self.residual_dropout_prob,
                             name="residual_dropout")
        result = (
            dropout(result, deterministic=not train) * self.residual_weight
            + unnormalized_inputs * self.input_weight
        )
        return result
