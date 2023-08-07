import tensorflow as tf
import math

HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]

# numeric_limits<float>::epsilon() 1.1920928955078125e-07
EPSILON = tf.constant(tf.keras.backend.epsilon(), dtype=tf.float32)
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001

def _next_power_of_2(x: int) -> int:
  r"""Returns the smallest power of 2 that is greater than x"""
  return 1 if x == 0 else 2 ** (x - 1).bit_length()

def inverse_mel_scale_scalar(mel_freq):
    return 700.0 * (tf.exp(mel_freq / 1127.0) - 1.0)

def inverse_mel_scale(mel_freq):
    return 700.0 * (tf.exp(mel_freq / 1127.0) - 1.0)

def mel_scale_scalar(freq):
    return 1127.0 * tf.math.log(1.0 + freq / 700.0)

def mel_scale(freq):
    return 1127.0 * tf.math.log(1.0 + freq / 700.0)

def _get_log_energy(strided_input: tf.Tensor, epsilon: tf.Tensor, energy_floor: float) -> tf.Tensor:
    r"""Returns the log energy of size (m) for a strided_input (m,*)"""
    dtype = strided_input.dtype
    log_energy = tf.math.log(tf.reduce_max(tf.reduce_sum(tf.pow(strided_input, 2), axis=1), epsilon))  # size (m)
    if energy_floor == 0.0:
        return log_energy
    return tf.maximum(log_energy, tf.constant(math.log(energy_floor), dtype=dtype))

def _feature_window_function(
    window_type: str,
    window_size: int,
    blackman_coeff: float,
) -> tf.Tensor:
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return tf.signal.hann_window(window_size, periodic=False, dtype=tf.float32)
    elif window_type == HAMMING:
        return tf.signal.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, dtype=tf.float32)
    elif window_type == POVEY:
        # like hanning but goes to zero at edges
        hann_window = tf.signal.hann_window(window_size, periodic=False, dtype=tf.float32)
        return tf.pow(hann_window, 0.85)
    elif window_type == RECTANGULAR:
        return tf.ones([window_size], dtype=tf.float32)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = tf.range(window_size, dtype=tf.float32)
        return (
            blackman_coeff
            - 0.5 * tf.cos(a * window_function)
            + (0.5 - blackman_coeff) * tf.cos(2 * a * window_function)
        )
    else:
        raise Exception("Invalid window type " + window_type)

def _get_strided(waveform, window_size, window_shift, snip_edges):
    assert waveform.shape.ndims == 1
    num_samples = tf.shape(waveform)[0]
    strides = (window_shift * waveform.shape[0], waveform.shape[0])

    if snip_edges:
        if num_samples < window_size:
            return tf.zeros((0, 0), dtype=waveform.dtype)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = tf.reverse(waveform, axis=[0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            pad_left = reversed_waveform[-pad:]
            waveform = tf.concat((pad_left, waveform, pad_right), axis=0)
        else:
            waveform = tf.concat((waveform[-pad:], pad_right), axis=0)

    sizes = (m, window_size)
    return tf.strided_slice(waveform, tf.zeros(2, dtype=tf.int32), tf.constant([m, window_size]), strides)

def _get_waveform_and_window_properties(
    waveform,
    channel,
    sample_frequency,
    frame_shift,
    frame_length,
    round_to_power_of_two,
    preemphasis_coefficient,
):
    channel = max(channel, 0)
    assert channel < waveform.shape[0], "Invalid channel {} for size {}".format(channel, waveform.shape[0])
    waveform = waveform[channel, :]  # size (n)
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= waveform.shape[0], "choose a window size {} that is [2, {}]".format(
        window_size, waveform.shape[0]
    )
    assert window_shift > 0, "`window_shift` must be greater than 0"
    assert padded_window_size % 2 == 0, (
        "the padded `window_size` must be divisible by two." " use `round_to_power_of_two` or change `frame_length`"
    )
    assert 0.0 <= preemphasis_coefficient <= 1.0, "`preemphasis_coefficient` must be between [0,1]"
    assert sample_frequency > 0, "`sample_frequency` must be greater than zero"
    return waveform, window_shift, window_size, padded_window_size

def _get_window(
    waveform,
    padded_window_size,
    window_size,
    window_shift,
    window_type,
    blackman_coeff,
    snip_edges,
    raw_energy,
    energy_floor,
    dither,
    remove_dc_offset,
    preemphasis_coefficient,
):
    epsilon = np.finfo(waveform.dtype.as_numpy_dtype).eps

    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        rand_gauss = tf.random.normal(strided_input.shape, dtype=waveform.dtype)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = tf.reduce_mean(strided_input, axis=1, keepdims=True)  # size (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        # Compute the log energy of each row/frame before applying preemphasis and window function
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m,)

    if preemphasis_coefficient != 0.0:
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = tf.pad(strided_input, [[0, 0], [1, 0]], mode="SYMMETRIC")[:, :-1]  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input

    # Apply window_function to each row/frame
    window_function = _feature_window_function(window_type, window_size, blackman_coeff, waveform.dtype)  # size (window_size,)
    window_function = tf.expand_dims(window_function, axis=0)  # size (1, window_size)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = tf.pad(strided_input, [[0, 0], [0, padding_right]], constant_values=0)

    # Compute energy after window function (not the raw one)
    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m,)

    return strided_input, signal_log_energy


def vtln_warp_freq(
    vtln_low_cutoff,
    vtln_high_cutoff,
    low_freq,
    high_freq,
    vtln_warp_factor,
    freq,
):
    l = vtln_low_cutoff * tf.maximum(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * tf.minimum(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l  # F(l)
    Fh = scale * h  # F(h)
    assert l > low_freq and h < high_freq

    # slope of left part of the 3-piece linear function
    scale_left = (Fl - low_freq) / (l - low_freq)
    # [slope of center part is just "scale"]
    # slope of right part of the 3-piece linear function
    scale_right = (high_freq - Fh) / (high_freq - h)

    res = tf.zeros_like(freq)

    outside_low_high_freq = tf.logical_or(tf.less(freq, low_freq), tf.greater(freq, high_freq))
    before_l = tf.less(freq, l)
    before_h = tf.less(freq, h)
    after_h = tf.greater_equal(freq, h)

    res = tf.where(after_h, high_freq + scale_right * (freq - high_freq), res)
    res = tf.where(before_h, scale * freq, res)
    res = tf.where(before_l, low_freq + scale_left * (freq - low_freq), res)
    res = tf.where(outside_low_high_freq, freq, res)

    return res
    
def vtln_warp_mel_freq(
    vtln_low_cutoff,
    vtln_high_cutoff,
    low_freq,
    high_freq,
    vtln_warp_factor,
    mel_freq,
):
    warped_freq = vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, inverse_mel_scale(mel_freq))
    warped_mel_freq = mel_scale(warped_freq)
    return warped_mel_freq
