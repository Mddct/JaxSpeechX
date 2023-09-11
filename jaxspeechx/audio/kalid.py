import tensorflow as tf
from typing import Tuple

import math

HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]

# numeric_limits<float>::epsilon() 1.1920928955078125e-07
EPSILON = tf.keras.backend.epsilon()
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001


def _next_power_of_2(x):
    x = tf.cast(x, dtype=tf.float32)
    return tf.cast(2**tf.math.ceil(tf.math.log(x) / tf.math.log(2.)),
                   dtype=tf.int32)


def _get_log_energy(strided_input: tf.Tensor, epsilon: tf.Tensor,
                    energy_floor: float) -> tf.Tensor:
    r"""Returns the log energy of size (m) for a strided_input (m,*)"""
    dtype = strided_input.dtype
    log_energy = tf.math.log(
        tf.math.maximum(tf.reduce_sum(tf.pow(strided_input, 2), axis=1),
                        epsilon))  # size (m)
    if energy_floor == 0.0:
        return log_energy
    return tf.math.maximum(log_energy,
                           tf.constant(math.log(energy_floor), dtype=dtype))


def _feature_window_function(
    window_type: str,
    window_size: int,
    blackman_coeff: float,
) -> tf.Tensor:
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return tf.signal.hann_window(window_size,
                                     periodic=False,
                                     dtype=tf.float32)
    elif window_type == HAMMING:
        return tf.signal.hamming_window(window_size,
                                        periodic=False,
                                        alpha=0.54,
                                        beta=0.46,
                                        dtype=tf.float32)
    elif window_type == POVEY:
        # like hanning but goes to zero at edges
        hann_window = tf.signal.hann_window(window_size,
                                            periodic=False,
                                            dtype=tf.float32)
        return tf.pow(hann_window, 0.85)
    elif window_type == RECTANGULAR:
        return tf.ones([window_size], dtype=tf.float32)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = tf.range(window_size, dtype=tf.float32)
        return (blackman_coeff - 0.5 * tf.cos(a * window_function) +
                (0.5 - blackman_coeff) * tf.cos(2 * a * window_function))
    else:
        raise Exception("Invalid window type " + window_type)


def _get_strided(waveform, window_size, window_shift, snip_edges):
    waveform = tf.expand_dims(waveform, axis=0)
    waveform_frames = tf.signal.frame(waveform,
                                      window_size,
                                      window_shift,
                                      pad_value=waveform[0][-1],
                                      pad_end=not snip_edges)

    waveform_frames = tf.squeeze(waveform_frames, axis=0)
    return waveform_frames


def _get_waveform_and_window_properties(
    waveform: tf.Tensor,
    channel: int,
    sample_frequency: tf.Tensor,
    frame_shift,
    frame_length,
    round_to_power_of_two: bool,
    preemphasis_coefficient,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    channel = max(channel, 0)
    tf.debugging.assert_less(channel,
                             tf.shape(waveform)[0],
                             message="Invalid channel size")
    waveform = waveform[channel, :]  # size (n)
    sample_frequency_f = tf.cast(sample_frequency, dtype=tf.float32)
    window_shift = tf.cast(
        sample_frequency_f * float(frame_shift) * MILLISECONDS_TO_SECONDS,
        tf.int32)
    window_size = tf.cast(
        sample_frequency_f * float(frame_length) * MILLISECONDS_TO_SECONDS,
        tf.int32)
    padded_window_size = _next_power_of_2(
        window_size) if round_to_power_of_two else window_size

    tf.debugging.assert_less_equal(2,
                                   window_size,
                                   message="choose a window size")
    tf.debugging.assert_less_equal(window_size,
                                   tf.shape(waveform),
                                   message="choose a window size")
    tf.debugging.assert_greater(window_shift,
                                0,
                                message="window_shift must be greater than 0")
    tf.debugging.assert_equal(
        padded_window_size % 2,
        0,
        message="the padded `window_size` must be divisible by two."
        " use `round_to_power_of_two` or change `frame_length`")
    assert 0.0 <= preemphasis_coefficient <= 1.0, "`preemphasis_coefficient` must be between [0,1]"
    tf.debugging.assert_greater(sample_frequency, 0,
                                "sample_frequency must be greater than zero")

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
    epsilon = EPSILON

    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift,
                                 snip_edges)

    if dither != 0.0:
        # NOTE(Mddct): tf.random.stateless_xxx is encouraged after tf2.11
        # TODO(Mddct): refiner later  using stateless_xxx with split generator
        rand_gauss = tf.random.normal(tf.shape(strided_input),
                                      dtype=waveform.dtype)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = tf.reduce_mean(strided_input, axis=1,
                                   keepdims=True)  # size (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        # Compute the log energy of each row/frame before applying preemphasis and window function
        signal_log_energy = _get_log_energy(strided_input, epsilon,
                                            energy_floor)  # size (m,)

    if preemphasis_coefficient != 0.0:
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = tf.pad(
            strided_input, [[0, 0], [1, 0]],
            mode="SYMMETRIC")[:, :-1]  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input

    # Apply window_function to each row/frame
    window_function = _feature_window_function(
        window_type,
        window_size,
        blackman_coeff,
    )  # size (window_size,)
    window_function = tf.expand_dims(window_function,
                                     axis=0)  # size (1, window_size)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = tf.pad(strided_input, [[0, 0], [0, padding_right]],
                               constant_values=0)

    # Compute energy after window function (not the raw one)
    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon,
                                            energy_floor)  # size (m,)

    return strided_input, signal_log_energy


def _subtract_column_mean(tensor: tf.Tensor, subtract_mean: bool) -> tf.Tensor:
    # subtracts the column mean of the tensor size (m, n) if subtract_mean=True
    # it returns size (m, n)
    if subtract_mean:
        col_means = tf.reduce_mean(tensor, axis=0, keepdims=True)
        tensor = tensor - col_means
    return tensor


def spectrogram(
    waveform: tf.Tensor,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    min_duration: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    window_type: str = "povey",
) -> tf.Tensor:
    r"""Create a spectrogram from a raw audio signal. This matches the input/output of Kaldi's
    compute-spectrogram-feats.

    Args:
        waveform (tf.Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at tf.keras.backend.epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``'povey'``)

    Returns:
        tf.Tensor: A spectrogram identical to what Kaldi would output. The shape is
        (m, ``padded_window_size // 2 + 1``) where m is calculated in _get_strided
    """
    epsilon = EPSILON

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length,
        round_to_power_of_two, preemphasis_coefficient)
    # NOTE(Mddct): tf graph mode using tf.function
    # TODO(Mddct): tf.cond to return short empty signal
    # if len(waveform) < min_duration * sample_frequency:
    #     # signal is too short
    #     return tf.constant([], dtype=tf.float32)

    strided_input, signal_log_energy = _get_window(
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
    )

    # Size (m, padded_window_size // 2 + 1, 2)
    fft = tf.signal.rfft(strided_input)

    # Convert the FFT into a power spectrum
    power_spectrum = tf.math.log(tf.maximum(
        tf.abs(fft)**2.0, epsilon))  # Size (m, padded_window_size // 2 + 1)
    power_spectrum = tf.concat(
        [tf.expand_dims(signal_log_energy, axis=1), power_spectrum[:, 1:]],
        axis=1)

    power_spectrum = _subtract_column_mean(power_spectrum, subtract_mean)
    return power_spectrum


def inverse_mel_scale_scalar(mel_freq):
    return 700.0 * (tf.exp(mel_freq / 1127.0) - 1.0)


def inverse_mel_scale(mel_freq):
    return 700.0 * (tf.exp(mel_freq / 1127.0) - 1.0)


def mel_scale_scalar(freq):
    return 1127.0 * tf.math.log(1.0 + freq / 700.0)


def mel_scale(freq):
    return 1127.0 * tf.math.log(1.0 + freq / 700.0)


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

    outside_low_high_freq = tf.logical_or(tf.less(freq, low_freq),
                                          tf.greater(freq, high_freq))
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
    warped_freq = vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq,
                                 high_freq, vtln_warp_factor,
                                 inverse_mel_scale(mel_freq))
    warped_mel_freq = mel_scale(warped_freq)
    return warped_mel_freq


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_low: float,
    vtln_high: float,
    vtln_warp_factor: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns:
        (tf.Tensor, tf.Tensor): The tuple consists of ``bins`` (which is
        melbank of size (``num_bins``, ``num_fft_bins``)) and ``center_freqs`` (which is
        center frequencies of bins of size (``num_bins``)).
    """

    tf.debugging.assert_greater(num_bins,
                                3,
                                message="Must have at least 3 mel bins")
    tf.debugging.assert_equal(window_length_padded % 2, 0)
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * tf.cast(sample_freq, dtype=tf.float32)

    if high_freq <= 0.0:
        high_freq = high_freq + nyquist
    low_freq = tf.convert_to_tensor(low_freq)

    tf.debugging.assert_less_equal(high_freq, nyquist)

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = tf.cast(sample_freq, dtype=tf.float32) / tf.cast(
        window_length_padded, dtype=tf.float32)
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / float(num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or (
        (low_freq < vtln_low < high_freq) and (0.0 < vtln_high < high_freq) and
        (vtln_low < vtln_high)
    ), "Bad values in options: vtln-low {} and vtln-high {}, versus " "low-freq {} and high-freq {}".format(
        vtln_low, vtln_high, low_freq, high_freq)

    bin = tf.expand_dims(tf.range(num_bins, dtype=tf.float32), axis=1)
    left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bin +
                                 1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bin +
                                2.0) * mel_freq_delta  # size(num_bins, 1)

    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq,
                                      vtln_warp_factor, left_mel)
        center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq,
                                        high_freq, vtln_warp_factor,
                                        center_mel)
        right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq,
                                       high_freq, vtln_warp_factor, right_mel)

    center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    mel = mel_scale(fft_bin_width *
                    tf.range(num_fft_bins, dtype=tf.float32))[tf.newaxis, :]

    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    if vtln_warp_factor == 1.0:
        # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
        bins = tf.maximum(tf.zeros((1, num_fft_bins), dtype=tf.float32),
                          tf.minimum(up_slope, down_slope))
    else:
        # warping can move the order of left_mel, center_mel, right_mel anywhere
        bins = tf.zeros_like(up_slope, dtype=tf.float32)
        up_idx = tf.logical_and(
            tf.greater(mel, left_mel),
            tf.less_equal(mel, center_mel))  # left_mel < mel <= center_mel
        down_idx = tf.logical_and(
            tf.greater(mel, center_mel),
            tf.less(mel, right_mel))  # center_mel < mel < right_mel
        bins = tf.where(up_idx, up_slope, bins)
        bins = tf.where(down_idx, down_slope, bins)

    return bins, center_freqs


def fbank(waveform,
          blackman_coeff=0.42,
          channel=-1,
          dither=0.0,
          energy_floor=1.0,
          frame_length=25.0,
          frame_shift=10.0,
          high_freq=0.0,
          htk_compat=False,
          low_freq=20.0,
          min_duration=0.0,
          num_mel_bins=23,
          preemphasis_coefficient=0.97,
          raw_energy=True,
          remove_dc_offset=True,
          round_to_power_of_two=True,
          sample_frequency=16000.0,
          snip_edges=True,
          subtract_mean=False,
          use_energy=False,
          use_log_fbank=True,
          use_power=True,
          vtln_high=-500.0,
          vtln_low=100.0,
          vtln_warp=1.0,
          window_type='povey'):
    waveform = tf.convert_to_tensor(waveform)
    sample_frequency = tf.convert_to_tensor(sample_frequency)
    num_mel_bins = tf.convert_to_tensor(num_mel_bins)
    dtype = waveform.dtype

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length,
        round_to_power_of_two, preemphasis_coefficient)

    # NOTE(Mddct): tf graph mode using tf.
    # TODO(Mddct): tf.cond to return short empty signal
    # if len(waveform) < min_duration * sample_frequency:
    #     # signal is too short
    #     return tf.constant([], dtype=dtype)
    # tf.debugging.assert_greater_equal(
    #     tf.cast(waveform.shape[0], dtype=tf.float32),
    #     min_duration * tf.cast(sample_frequency, dtype=tf.float32))

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input, signal_log_energy = _get_window(
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
    )

    # NOTE(Mddct): tf.signal.rfft is slow than torch fft
    # size (m, padded_window_size // 2 + 1)
    # def fast_rfft(input: tf.Tensor):
    #     input_dl = tf.experimental.dlpack.to_dlpack(input)
    #     input_jax = jax.dlpack.from_dlpack(input_dl)
    #     output_jax = jax.numpy.fft.rfft(input_jax)
    #     # output_jax = jax.numpy.abs(output_jax)
    #     # if use_power:
    #     #     output_jax = jax.lax.pow(output_jax, 2.0)
    #     out_dl = jax.dlpack.to_dlpack(output_jax)
    #     out_tf = tf.experimental.dlpack.from_dlpack(out_dl)
    #     return out_tf

    # spectrum = tf.abs(
    #     tf.py_function(fast_rfft, inp=[strided_input], Tout=tf.complex64))
    spectrum = tf.abs(tf.signal.rfft(strided_input))
    if use_power:
        spectrum = tf.pow(spectrum, 2.0)

    # size (num_mel_bins, padded_window_size // 2)
    mel_energies, _ = get_mel_banks(num_mel_bins, padded_window_size,
                                    sample_frequency, low_freq, high_freq,
                                    vtln_low, vtln_high, vtln_warp)
    mel_energies = tf.cast(mel_energies, dtype=dtype)

    # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = tf.pad(mel_energies, [[0, 0], [0, 1]], constant_values=0)

    # sum with mel filterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = tf.matmul(spectrum, tf.transpose(mel_energies))
    if use_log_fbank:
        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = tf.math.log(tf.maximum(mel_energies, EPSILON))

    # if use_energy then add it as the last column for htk_compat == true else first column
    if use_energy:
        signal_log_energy = tf.expand_dims(signal_log_energy,
                                           axis=1)  # size (m, 1)
        # returns size (m, num_mel_bins + 1)
        if htk_compat:
            mel_energies = tf.concat([mel_energies, signal_log_energy], axis=1)
        else:
            mel_energies = tf.concat([signal_log_energy, mel_energies], axis=1)

    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies


def _get_dct_matrix(num_ceps: int, num_mel_bins: int) -> tf.Tensor:
    # Create DCT basis
    basis = tf.signal.dct(tf.eye(num_mel_bins), type=2, norm='ortho')
    # Kaldi expects the first cepstral to be a weighted sum of factor sqrt(1/num_mel_bins)
    basis = tf.tensor_scatter_nd_update(
        basis, [[i, 0] for i in range(num_mel_bins)],
        [math.sqrt(1.0 / num_mel_bins)] * num_mel_bins)

    # We only keep num_ceps coefficients
    basis = basis[:, :num_ceps]

    return basis


def _get_lifter_coeffs(num_ceps: int, cepstral_lifter: float) -> tf.Tensor:
    """Returns a liftering coefficients of size (num_ceps).

    Args: 
        num_ceps: The number of cepstral coefficients.
        cepstral_lifter: The cepstral lifter coefficient.

    Returns:
        A liftering coefficients of size (num_ceps).
    """

    i = tf.range(num_ceps, dtype=tf.float32)
    return 1.0 + 0.5 * cepstral_lifter * tf.sin(math.pi * i / cepstral_lifter)


def mfcc(
    waveform,
    blackman_coeff=0.42,
    cepstral_lifter=22.0,
    channel=-1,
    dither=0.0,
    energy_floor=1.0,
    frame_length=25.0,
    frame_shift=10.0,
    high_freq=0.0,
    htk_compat=False,
    low_freq=20.0,
    num_ceps=13,
    min_duration=0.0,
    num_mel_bins=23,
    preemphasis_coefficient=0.97,
    raw_energy=True,
    remove_dc_offset=True,
    round_to_power_of_two=True,
    sample_frequency=16000.0,
    snip_edges=True,
    subtract_mean=False,
    use_energy=False,
    vtln_high=-500.0,
    vtln_low=100.0,
    vtln_warp=1.0,
    window_type="povey",
):
    assert num_ceps <= num_mel_bins, "num_ceps cannot be larger than num_mel_bins: %d vs %d" % (
        num_ceps, num_mel_bins)

    # The mel_energies should not be squared (use_power=True), not have mean subtracted
    # (subtract_mean=False), and use log (use_log_fbank=True).
    # size (m, num_mel_bins + use_energy)
    feature = fbank(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        channel=channel,
        dither=dither,
        energy_floor=energy_floor,
        frame_length=frame_length,
        frame_shift=frame_shift,
        high_freq=high_freq,
        htk_compat=htk_compat,
        low_freq=low_freq,
        min_duration=min_duration,
        num_mel_bins=num_mel_bins,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        round_to_power_of_two=round_to_power_of_two,
        sample_frequency=sample_frequency,
        snip_edges=snip_edges,
        subtract_mean=False,
        use_energy=use_energy,
        use_log_fbank=True,
        use_power=True,
        vtln_high=vtln_high,
        vtln_low=vtln_low,
        vtln_warp=vtln_warp,
        window_type=window_type,
    )

    if use_energy:
        # size (m)
        signal_log_energy = feature[:, num_mel_bins if htk_compat else 0]
        # offset is 0 if htk_compat==True else 1
        mel_offset = int(not htk_compat)
        feature = feature[:, mel_offset:(num_mel_bins + mel_offset)]

    # size (num_mel_bins, num_ceps)
    dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins)
    # size (m, num_ceps)
    feature = tf.matmul(feature, dct_matrix)
    if cepstral_lifter != 0.0:
        # size (1, num_ceps)
        lifter_coeffs = _get_lifter_coeffs(num_ceps, cepstral_lifter)[None, :]
        feature *= lifter_coeffs

    # if use_energy then replace the last column for htk_compat == true else first column
    if use_energy:
        feature[:, 0] = signal_log_energy

    if htk_compat:
        energy = feature[:, 0][:, tf.newaxis]  # size (m, 1)
        feature = feature[:, 1:]  # size (m, num_ceps - 1)
        if not use_energy:
            # scale on C0 (actually removing a scale we previously added that's
            # part of one common definition of the cosine transform.)
            energy *= math.sqrt(2)

        feature = tf.concat((feature, energy), axis=1)

    feature = _subtract_column_mean(feature, subtract_mean)
    return feature
