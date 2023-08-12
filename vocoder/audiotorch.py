import math
import numpy as np
import torch
import torchaudio
import vocoder.hparams as hp
from scipy.signal import lfilter
import soundfile as sf

# Function to convert labels to float values
def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.

# Function to convert float values to labels
def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)

# Load audio using torchaudio
def load_wav(path):
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform[0].numpy()

# Save audio using torchaudio
def save_wav(x, path):
    sf.write(path, x.astype(np.float32), hp.sample_rate)

# Function to split signal into coarse and fine parts
def split_signal(x):
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

# Function to combine coarse and fine parts of a signal
def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15

# Encoding function for 16-bit audio
def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

# Mel basis for mel spectrogram
mel_basis = None

# Function to convert linear spectrogram to mel spectrogram
def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return torch.matmul(torch.tensor(mel_basis), torch.tensor(spectrogram))

# Function to build mel basis
def build_mel_basis():
    return torchaudio.transforms.MelSpectrogram(sample_rate=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels)(torch.zeros(hp.n_fft))

# Normalize the spectrogram
def normalize(S):
    return torch.clamp((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

# Denormalize the spectrogram
def denormalize(S):
    return (torch.clamp(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

# Convert amplitude to decibels
def amp_to_db(x):
    return 20 * torch.log10(torch.max(torch.tensor(1e-5), x))

# Convert decibels to amplitude
def db_to_amp(x):
    return torch.pow(10.0, x * 0.05)

# Compute spectrogram using torchaudio
def spectrogram(y):
    D = stft(y)
    S = amp_to_db(torch.abs(D)) - hp.ref_level_db
    return normalize(S)

# Compute mel spectrogram using torchaudio
def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(torch.abs(D)))
    return normalize(S)

# Compute Short-Time Fourier Transform using torchaudio
def stft(y):
    return torchaudio.transforms.MelSpectrogram(sample_rate=hp.sample_rate, n_fft=hp.n_fft, hop_length=hp.hop_length)(torch.tensor(y))

# Apply pre-emphasis filter
def pre_emphasis(x):
    b = torch.tensor([1, -hp.preemphasis], dtype=torch.float32)
    a = torch.tensor([1], dtype=torch.float32)
    return lfilter(b.numpy(), a.numpy(), x)

# Apply de-emphasis filter
def de_emphasis(x):
    b = torch.tensor([1], dtype=torch.float32)
    a = torch.tensor([1, -hp.preemphasis], dtype=torch.float32)
    return lfilter(b.numpy(), a.numpy(), x)

# Encoding function for mu-law compression
def encode_mu_law(x, mu):
    mu = mu - 1
    fx = torch.sign(x) * torch.log(1 + mu * torch.abs(x)) / torch.log(1 + mu)
    return torch.floor((fx + 1) / 2 * mu + 0.5)

# Decoding function for mu-law compression
def decode_mu_law(y, mu, from_labels=True):
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = torch.sign(y) / mu * ((1 + mu) ** torch.abs(y) - 1)
    return x
