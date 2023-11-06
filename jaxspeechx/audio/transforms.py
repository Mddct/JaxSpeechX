from typing import Optional, Tuple
import tensorflow as tf
import math


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
    num_wavs, length = tf.shape(waveform)[0], tf.shape(waveform)[1]
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
        lambda: waveform,
        _resample,
    )


def add_noise(waveform: tf.Tensor,
              noise: tf.Tensor,
              snr: tf.Tensor,
              lengths: Optional[tf.Tensor] = None) -> tf.Tensor:
    L = tf.shape(waveform)[-1]
    L_noise = tf.shape(noise)[-1]

    def length_ge(noise):
        random_start = tf.random.uniform(shape=(),
                                         maxval=L - L_noise + 1,
                                         dtype=tf.int32)
        noise = tf.pad(noise,
                       [[0, 0], [random_start, L - L_noise - random_start]])
        return noise

    def length_lt(noise):
        random_start = tf.random.uniform(shape=(),
                                         maxval=L_noise - L + 1,
                                         dtype=tf.int32)
        return noise[..., random_start:random_start + L]

    noise = tf.cond(tf.greater(L, L_noise), lambda: length_ge(noise),
                    lambda: length_lt(noise))
    # compute scale
    if lengths is not None:
        mask = tf.expand_dims(tf.range(0, L, dtype=lengths.dtype),
                              axis=0) < tf.expand_dims(lengths, axis=-1)
        masked_waveform = waveform * tf.cast(mask, dtype=waveform.dtype)
        masked_noise = noise * tf.cast(mask, dtype=noise.dtype)
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = tf.norm(masked_waveform, ord=2, axis=-1)**2
    energy_noise = tf.norm(masked_noise, ord=2, axis=-1)**2
    original_snr_db = 10 * (tf.math.log(energy_signal) -
                            tf.math.log(energy_noise)) / tf.math.log(10.0)
    scale = 10**((original_snr_db - snr) / 20.0)

    # scale noise
    scaled_noise = tf.expand_dims(scale, axis=-1) * noise

    return waveform + scaled_noise


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
        out_lengths = tf.cast(tf.math.ceil(lengths * target_sample_rate /
                                           source_sample_rate),
                              dtype=lengths.dtype)

    return resample(waveform, source_sample_rate,
                    target_sample_rate), out_lengths


def _apply_convolve_mode(conv_result, x_length, y_length, mode):
    valid_convolve_modes = ["full", "valid", "same"]
    if mode == "full":
        return conv_result
    elif mode == "valid":
        target_length = tf.math.maximum(x_length, y_length) - tf.math.minimum(
            x_length, y_length) + 1
        start_idx = (tf.shape(conv_result)[-1] - target_length) // 2
        return conv_result[..., start_idx:start_idx + target_length]
    elif mode == "same":
        start_idx = (tf.shape(conv_result)[-1] - x_length) // 2
        return conv_result[..., start_idx:start_idx + x_length]
    else:
        raise ValueError(
            f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}."
        )


def convolve(x, y, mode="full"):

    x_size, y_size = tf.shape(x)[-1], tf.shape(y)[-1]
    x, y = tf.cond(x_size < y_size, lambda: (y, x), lambda: (x, y))
    x_size, y_size = tf.shape(x)[-1], tf.shape(y)[-1]

    def reshape_tensors(x, y):
        new_shape = tf.math.maximum(tf.shape(x)[:-1], tf.shape(y)[:-1])
        x = tf.broadcast_to(x, tf.concat([new_shape, [x_size]], axis=0))
        y = tf.broadcast_to(y, tf.concat([new_shape, [y_size]], axis=0))
        return x, y

    x, y = tf.cond(x_size != y_size, lambda: reshape_tensors(x, y), lambda:
                   (x, y))
    x_size, y_size = tf.shape(x)[-1], tf.shape(y)[-1]

    num_signals = tf.reduce_prod(tf.shape(x)[:-1])
    reshaped_x = tf.reshape(x, (tf.cast(num_signals, tf.int32), x_size))
    reshaped_y = tf.reshape(y, (tf.cast(num_signals, tf.int32), y_size))

    reshaped_weight = tf.reverse(tf.expand_dims(reshaped_y, axis=-1), axis=[1])
    reshaped_weight = tf.transpose(reshaped_weight, [1, 0, 2])
    padding_nums = tf.shape(reshaped_y)[-1] - 1
    zeros = tf.zeros(
        tf.concat([tf.shape(reshaped_y)[:-1], [padding_nums]], axis=-1))
    reshaped_x = tf.concat([zeros, reshaped_x, zeros], axis=-1)
    # TODO: groups conv1d to support multiple channel
    output = tf.nn.conv1d(
        input=tf.expand_dims(reshaped_x, axis=-1),
        filters=reshaped_weight,
        stride=1,
        padding='VALID',
        data_format='NWC',
    )
    output_shape = tf.concat([tf.shape(x)[:-1], [-1]], axis=0)
    result = tf.reshape(output, output_shape)
    return _apply_convolve_mode(result, x_size, y_size, mode)


def add_rir(waveform: tf.Tensor, rir: tf.Tensor, norm_rir: bool = False):
    if norm_rir:
        norm_rir = rir - tf.math.reduce_mean(rir, axis=-1)
    else:
        norm_rir = rir
    size = tf.shape(waveform)[-1]
    waveform = convolve(waveform, norm_rir, 'full')
    waveform = waveform[..., :size]
    # normalize data samples to [-1,1] after rir convolution to avoid nans with fp16 training
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/perturb.py#L400
    waveform = waveform / tf.reduce_max(tf.math.abs(waveform), axis=1)
    return waveform
