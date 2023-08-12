import struct
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad = None

int16_max = (2 ** 15) - 1

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.
    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform's sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = torchaudio.load(str(fpath_or_wav), num_frames=-1)
        wav = wav[0]  # Extract the mono channel
        source_sr = source_sr.item()
    else:
        wav = torch.tensor(fpath_or_wav)

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        resample = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=sampling_rate)
        wav = resample(wav)

    # Apply the preprocessing: normalize volume and shorten long silences
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)

    return wav

def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this is not a log-mel spectrogram.
    """
    mel_transform = MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    frames = mel_transform(wav)
    return frames.transpose(0, 1).numpy()

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a torch tensor
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = (wav * int16_max).clamp_(-int16_max, int16_max).to(torch.int16)
    pcm_wave = pcm_wave.numpy().tobytes()

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        pcm_window = pcm_wave[window_start * 2:window_end * 2]
        voice_flags.append(vad.is_speech(pcm_window, sample_rate=sampling_rate))

    voice_flags = np.array(voice_flags)
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    return wav[audio_mask > 0]


# normalize volume funcation
def normalize_volume(wav, target_dBFS, increase_only=True):
    rms = np.sqrt(np.mean(wav ** 2))
    factor = 10 ** ((target_dBFS - 20 * np.log10(rms)) / 20)

    if increase_only:
        factor = max(1, factor)

    normalized_wav = wav * factor
    return normalized_wav


# Add any missing functions like `normalize_volume` and define constants from encoder.params_data if needed.

# Example usage:
# wav = preprocess_wav("path_to_audio.wav")
# mel_spec = wav_to_mel_spectrogram(wav)
# trimmed_wav = trim_long_silences(wav)
