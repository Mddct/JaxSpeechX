from typing import Optional, Tuple
import tensorflow as tf
import math

from tensorflow.python.ops.array_ops import lower_bound


def _get_sinc_resample_kernel(
    orig_freq: tf.Tensor,
    new_freq: tf.Tensor,
    gcd: tf.Tensor,
    lowpass_filter_width: float = 6.,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    dtype: Optional[tf.dtypes.DType] = None,
):
    # NOTE: orig_freq.dtype == new_freq.dtype, tf.int32 or tf.int64
    if resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError(
            "Invalid resampling method: {}".format(resampling_method))
    orig_freq = tf.cast(orig_freq, dtype=tf.int32) // gcd
    new_freq = tf.cast(new_freq, dtype=tf.int32) // gcd
    orig_freq = tf.cast(orig_freq, dtype=dtype)
    new_freq = tf.cast(new_freq, dtype=dtype)
    rolloff = tf.cast(rolloff, dtype=dtype)
    lowpass_filter_width = tf.cast(lowpass_filter_width, dtype=dtype)
    base_freq = tf.math.minimum(orig_freq, new_freq)
    base_freq = base_freq * rolloff

    width = tf.math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else tf.float32

    idx = tf.range(-width, width + orig_freq,
                   dtype=idx_dtype)[None, None] / orig_freq

    t = tf.range(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq + idx
    t *= base_freq
    t = tf.clip_by_value(t, -lowpass_filter_width, lowpass_filter_width)
    if resampling_method == "sinc_interp_hann":
        window = tf.cos(t * math.pi / lowpass_filter_width / 2)**2
    else:
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = tf.constant(float(beta))
        window = tf.math.i0(beta_tensor * tf.math.sqrt(
            1 - (t / lowpass_filter_width)**2)) / tf.math.i0(beta_tensor)

    t = t * math.pi

    scale = base_freq / orig_freq
    kernels = tf.where(t == 0, tf.constant(1.0, dtype=dtype), tf.sin(t) / t)
    kernels *= window * scale

    if dtype is None:
        kernels = tf.cast(kernels, tf.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: tf.Tensor,
    orig_freq: tf.Tensor,
    new_freq: tf.Tensor,
    gcd: tf.Tensor,
    kernel: tf.Tensor,
    width: tf.Tensor,
):

    if not tf.dtypes.as_dtype(waveform.dtype).is_floating:
        raise TypeError(
            f"Expected floating point type for waveform tensor, but received {waveform.dtype}."
        )

    orig_freq = tf.cast(orig_freq, dtype=tf.int32) // gcd
    new_freq = tf.cast(new_freq, dtype=tf.int32) // gcd

    # pack batch
    shape = tf.concat([tf.constant([-1]), tf.shape(waveform)[-1:]], axis=0)
    waveform = tf.reshape(waveform, shape)
    shape = tf.shape(waveform)
    num_wavs, length = waveform.shape
    # return waveform
    waveform = tf.pad(waveform, ((0, 0), (width, width + orig_freq)))

    # kernel = tf.expand_dims(kernel, axis=0)
    kernel = tf.transpose(kernel, [1, 0, 2])
    src = tf.signal.frame(waveform,
                          frame_length=tf.shape(kernel)[-1],
                          frame_step=orig_freq,
                          axis=-1)  # [..., num_frames, frame_length]
    src = tf.expand_dims(src, axis=2)
    kernel = tf.expand_dims(kernel, axis=1)

    dst = src * kernel
    resampled = tf.math.reduce_sum(dst, axis=-1, keepdims=False)
    resampled = tf.reshape(resampled, [num_wavs, -1])
    target_length = tf.cast(tf.math.ceil(new_freq * length / orig_freq),
                            tf.int32)
    resampled = resampled[..., :target_length]

    # unpack batch
    shape = tf.concat([shape[:-1], tf.shape(resampled)[-1:]], axis=0)

    # exit(1)
    resampled = tf.reshape(resampled, shape)
    return resampled


def resample(waveform: tf.Tensor,
             orig_freq: tf.Tensor,
             new_freq: tf.Tensor,
             lowpass_filter_width=6,
             rolloff=0.99,
             resampling_method="sinc_interp_hann",
             beta=None):

    tf.debugging.assert_greater(orig_freq, 0)
    tf.debugging.assert_greater(new_freq, 0)

    # If the original and desired frequencies are the same, just return the original waveform.
    def _resample():
        gcd = tf.experimental.numpy.gcd(orig_freq, new_freq)
        # Create the sinc resample kernel.

        kernel, width = _get_sinc_resample_kernel(
            orig_freq,
            new_freq,
            gcd,
            lowpass_filter_width,
            rolloff,
            resampling_method,
            beta,
            dtype=waveform.dtype,
        )
        width = tf.cast(width, dtype=tf.int32)

        # Resample the waveform.
        resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq,
                                                gcd, kernel, width)
        return resampled

    return tf.cond(
        orig_freq == new_freq,
        # lambda: waveform,
        _resample,
        _resample,
    )


def speed(
    waveform: tf.Tensor,
    orig_freq: tf.Tensor,
    factor: tf.Tensor,
    lengths: Optional[tf.Tensor] = None
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    source_sample_rate = tf.cast(factor *
                                 tf.cast(orig_freq, dtype=factor.dtype),
                                 dtype=orig_freq.dtype)
    target_sample_rate = orig_freq

    gcd = tf.experimental.numpy.gcd(source_sample_rate, target_sample_rate)
    source_sample_rate = source_sample_rate // gcd
    target_sample_rate = target_sample_rate // gcd

    if lengths is None:
        out_lengths = None
    else:
        out_lengths = tf.math.ceil(lengths * target_sample_rate /
                                   source_sample_rate)

    return resample(waveform, source_sample_rate,
                    target_sample_rate), out_lengths
