import numpy as np
import torch
from scipy.io import wavfile
import soundfile as sf
import torchaudio

def load_wav(path, sr):
    return torchaudio.load(path, num_frames=-1)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, torch.max(torch.abs(wav)))
    wavfile.write(path, sr, wav.numpy().astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    sf.write(path, wav.numpy().astype(np.float32), sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return torch.cat([wav[:1], wav[1:] - k * wav[:-1]], dim=0)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return torch.cat([wav[:1], wav[1:] + k * wav[:-1]], dim=0)
    return wav

def start_and_end_indices(quantized, silence_threshold=2):
    quantized = quantized.numpy()  # Convert to NumPy array
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break
    
    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold
    
    return start, end

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def linearspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(torch.abs(D), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def melspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(torch.abs(D), hparams), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S



def inv_linear_spectrogram(linear_spectrogram, hparams):
    """Converts linear spectrogram to waveform using torchaudio"""
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram
    
    S = _db_to_amp(D + hparams.ref_level_db)  # Convert back to linear
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.cpu().numpy().T ** hparams.power)
        y = processor.istft(D)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def inv_mel_spectrogram(mel_spectrogram, hparams):
    """Converts mel spectrogram to waveform using torchaudio"""
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram
    
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.cpu().numpy().T ** hparams.power)
        y = processor.istft(D)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")

def _griffin_lim(S, hparams):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = torch.exp(2j * np.pi * torch.rand(*S.shape))
    S_complex = torch.abs(S).to(torch.complex64)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = torch.exp(1j * torch.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return torch.stft(y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size, center=False).transpose(1, 2)

def _istft(y, hparams):
    return torch.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size, center=False)



def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M

def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r

def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectrogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return torch.mm(_mel_basis, spectrogram)

def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = torch.pinverse(_build_mel_basis(hparams))
    return torch.clamp_min(torch.mm(_inv_mel_basis, mel_spectrogram), 1e-10)

def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=hparams.sample_rate,
        n_fft=hparams.n_fft,
        f_min=hparams.fmin,
        f_max=hparams.fmax,
        n_mels=hparams.num_mels
    )

def _amp_to_db(x, hparams):
    min_level = torch.exp(torch.tensor(hparams.min_level_db / 20 * np.log(10), dtype=torch.float32))
    return 20 * torch.log10(torch.max(min_level, x))

def _db_to_amp(x):
    return torch.pow(10.0, x * 0.05)

def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return torch.clamp(
                (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                -hparams.max_abs_value,
                hparams.max_abs_value
            )
        else:
            return torch.clamp(
                hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)),
                0,
                hparams.max_abs_value
            )
    else:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
        if hparams.symmetric_mels:
            return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
        else:
            return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))


def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((torch.clamp(D, -hparams.max_abs_value, hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((torch.clamp(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
    
    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
