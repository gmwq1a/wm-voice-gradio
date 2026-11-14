#!/usr/bin/env python3
# Gemini TTS 干声 DSP 全效果库 - 纯 numpy/scipy（Colab 零安装）
# 用法: python3 voice_dsp_v2.py 输入.wav 输出.wav 效果名
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

def read_wav(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(axis=1)
    return sr, x / (np.max(np.abs(x)) or 1.0)

def write_wav(path, sr, x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1, 1)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))

def bandpass(x, sr, lo, hi, order=4):
    b, a = signal.butter(order, [lo/(sr/2), hi/(sr/2)], btype='band')
    return signal.lfilter(b, a, x)
def lowpass(x, sr, fc, order=4):
    b, a = signal.butter(order, fc/(sr/2), btype='low')
    return signal.lfilter(b, a, x)
def highpass(x, sr, fc, order=4):
    b, a = signal.butter(order, fc/(sr/2), btype='high')
    return signal.lfilter(b, a, x)
def low_shelf_boost(x, sr, fc=200, gain=0.5):
    b, a = signal.butter(2, fc/(sr/2), btype='low')
    return x + gain * signal.lfilter(b, a, x)
def softclip(x, drive=2.0, out=0.8):
    return out * np.tanh(drive * x)
def norm(x, peak=0.85):
    return x / (np.max(np.abs(x)) or 1.0) * peak

def _fftconvolve_chunked(x, ir, block):
    n_ir = len(ir)
    out = np.zeros(len(x), dtype=np.float64)
    for s in range(0, len(x), block):
        seg = x[s:s+block]
        c = signal.fftconvolve(seg, ir)
        e = min(len(x), s + len(c))
        out[s:e] += c[:e-s]
    return out

def reverb(x, sr, decay_s=3.5, wet=0.45, seed=7):
    n = int(sr * decay_s)
    t = np.arange(n) / sr
    ir = np.random.default_rng(seed).standard_normal(n) * np.exp(-6.9 * t / decay_s)
    ir[0] = 1.0
    if len(x) > sr * 60:
        w = _fftconvolve_chunked(x, ir, int(sr * 20))[:len(x)]
    else:
        w = signal.fftconvolve(x, ir)[:len(x)]
    return (1-wet)*x + wet * norm(w, 1.0)

def echo(x, sr, delays_ms, gains):
    out = x.copy()
    for d, g in zip(delays_ms, gains):
        n = int(d * sr / 1000)
        if n < len(x): out[n:] += g * x[:-n]
    return out

def modulated_delay(x, sr, base_ms, depth_ms, rate_hz, mix=0.5, feedback=0.0):
    """调制延迟线：chorus/flanger 的核心"""
    n = len(x)
    t = np.arange(n) / sr
    delay = (base_ms + depth_ms * np.sin(2 * np.pi * rate_hz * t)) / 1000.0
    idx = np.clip(np.arange(n) - delay * sr, 0, n - 1)
    wet = np.interp(idx, np.arange(n), x)
    out = x.copy()
    if feedback > 0:  # flanger 带回授
        for i in range(1, 6):
            out += (feedback ** i) * np.roll(wet, i * int(base_ms * sr / 1000))
    return (1 - mix) * x + mix * (wet + feedback * out) / (1 + feedback)

def wow_flutter(x, sr, rate=3.5, depth_ms=1.5):
    n = len(x); t = np.arange(n) / sr
    warp = t + (depth_ms/1000.0) * np.sin(2*np.pi*rate*t) / (2*np.pi*rate)
    return np.interp(warp, t, x)

def pitch_shift(x, semis):
    """重采样变调（时长会变，跟磁带变速一个道理）"""
    r = 2 ** (semis/12.0)
    return signal.resample(x, int(len(x)/r))

def detune(x, cents):
    r = 2 ** (cents/1200.0)
    y = signal.resample(x, int(len(x)/r))
    return np.interp(np.linspace(0, len(y)-1, len(x)), np.arange(len(y)), y)

def compressor(x, sr, thresh=0.25, ratio=6.0, attack_ms=5, release_ms=80):
    env = np.abs(x)
    k_a = np.exp(-1/(sr*attack_ms/1000)); k_r = np.exp(-1/(sr*release_ms/1000))
    sm = np.zeros_like(env); e = 0.0
    for i, v in enumerate(env):
        e = max(v, e*(k_a if v > e else k_r)); sm[i] = e
    gain = np.where(sm > thresh, (thresh*(sm/thresh)**(1/ratio))/np.maximum(sm, 1e-9), 1.0)
    return norm(x * gain)

def autowah(x, sr, block=512, f_lo=300, f_hi=1800):
    env = lowpass(np.abs(x), sr, 8)
    env /= (np.max(env) or 1.0)
    out = np.zeros_like(x)
    hop = block // 2
    win = np.hanning(block)
    for s in range(0, len(x)-block, hop):
        fc = f_lo + (f_hi-f_lo) * env[s+hop]
        lo, hi = max(fc*0.6, 60), min(fc*1.6, sr/2-100)
        b, a = signal.butter(2, [lo/(sr/2), hi/(sr/2)], btype='band')
        out[s:s+block] += signal.lfilter(b, a, x[s:s+block]) * win
    return norm(out)

def heartbeat_track(n, sr, bpm=60, vol=0.25):
    t = np.arange(n) / sr
    out = np.zeros(n)
    period = 60.0 / bpm
    for t0 in np.arange(0, t[-1], period):
        for off, v in [(0, 1.0), (0.28, 0.6)]:
            i0 = int((t0+off)*sr); m = int(0.18*sr)
            if i0+m <= n:
                tt = np.arange(m)/sr
                out[i0:i0+m] += v * np.sin(2*np.pi*55*tt) * np.exp(-tt/0.06)
    breath = np.random.default_rng(3).standard_normal(n)
    breath = lowpass(breath, sr, 500)
    lfo = 0.5 + 0.5*np.sin(2*np.pi*0.25*t)
    return vol*out + 0.05*breath*lfo

def fx(name, x, sr):
    rng = np.random.default_rng(42)
    N = len(x)
    noise = lambda lo, hi: bandpass(rng.standard_normal(N), sr, lo, hi)
    if name == 'room':       return reverb(x, sr, 0.35, 0.22)
    if name == 'corridor':   return norm(echo(x, sr, [90, 180, 270], [0.35, 0.22, 0.12]) + 0.3*reverb(x, sr, 1.2, 0.5))
    if name == 'cockpit':    return norm(echo(bandpass(x, sr, 300, 4000), sr, [25, 50], [0.3, 0.15]) + 0.05*noise(40, 120))
    if name == 'close':      return low_shelf_boost(x, sr, 220, 0.8)
    if name == 'far':        return norm(0.25*lowpass(x, sr, 2000) + 0.6*reverb(lowpass(x, sr, 2000), sr, 2.5, 1.0))
    if name == 'submarine':
        t = np.arange(N)/sr
        return norm(lowpass(x, sr, 1500) + 0.12*np.sin(2*np.pi*40*t) + 0.06*lowpass(rng.standard_normal(N), sr, 80))
    if name == 'temple':
        air = highpass(x, sr, 6000)
        d = detune(x, 5); d2 = detune(x, -5)
        return norm(reverb(x, sr, 4.5, 0.5) + 0.2*reverb(d, sr, 4.5, 0.6) + 0.2*reverb(d2, sr, 4.5, 0.6) + 0.5*air)
    if name == 'phone':      return norm(softclip(bandpass(x, sr, 300, 3400), 2.5) + 0.02*noise(300, 3400))
    if name == 'radio40':    return norm(softclip(bandpass(x, sr, 150, 4500), 2.0) + 0.03*noise(200, 6000))
    if name == 'gramophone': return norm(wow_flutter(bandpass(x, sr, 500, 4000), sr) + 0.04*noise(1000, 6000))
    if name == 'megaphone':  return norm(echo(softclip(bandpass(x, sr, 500, 4000), 4.0), sr, [120, 240], [0.3, 0.18]))
    if name == 'underwater':
        t = np.arange(N)/sr
        return norm(lowpass(reverb(x, sr, 1.5, 0.5), sr, 800) * (0.8 + 0.2*np.sin(2*np.pi*0.5*t)))
    if name == 'wall':       return norm(low_shelf_boost(lowpass(x, sr, 500), sr, 300, 1.0))
    if name == 'tv':         return norm(softclip(bandpass(x, sr, 100, 8000), 1.8))
    if name == 'cassette':   return norm(wow_flutter(lowpass(x, sr, 10000), sr, 2.5, 1.0) + 0.02*noise(3000, 9000))
    if name == 'kid':        return pitch_shift(x, 4)
    if name == 'giant':      return pitch_shift(x, -6)
    if name == 'ringmod':
        t = np.arange(N)/sr
        return norm(x * np.sin(2*np.pi*45*t))
    if name == 'chipmunk':   return pitch_shift(x, 7)
    if name == 'demon':      return norm(reverb(softclip(pitch_shift(x, -5), 2.0), sr, 2.0, 0.35))
    if name == 'harmony':    return norm(x + 0.6*detune(x, 8) + 0.6*detune(x, -8))
    if name == 'vibrato':    return wow_flutter(x, sr, 6.0, 4.0)
    if name == 'tremolo':
        t = np.arange(N)/sr
        return x * (0.5 + 0.5*np.sin(2*np.pi*6*t))
    if name == 'phaser':
        t = np.arange(N)/sr
        a = 0.7 + 0.25*np.sin(2*np.pi*0.4*t)
        y = np.zeros_like(x); xm1 = 0.0; ym1 = 0.0
        for i in range(N):
            ym1 = -a[i]*x[i] + xm1 + a[i]*ym1
            y[i] = ym1; xm1 = x[i]
        return norm(x + y)
    if name == 'pingpong':
        L = echo(x, sr, [250, 750], [0.5, 0.3]); R = echo(x, sr, [500, 1000], [0.5, 0.3])
        return np.stack([norm(L, 0.7), norm(R, 0.7)], axis=1)
    if name == 'autowah':    return autowah(x, sr)
    if name == 'radioloud':  return compressor(x, sr)
    if name == 'reverse':    return x[::-1]
    if name == 'slow':       return signal.resample(x, int(N/0.8))
    if name == 'bit8':
        y = np.round(x*127)/127
        return signal.resample(y, int(len(y)*8000/sr))
    if name == 'binaural':
        t = np.arange(N)/sr
        pan = 0.5 + 0.5*np.sin(2*np.pi*0.33*t)
        return np.stack([x*np.sqrt(1-pan), x*np.sqrt(pan)], axis=1)
    if name == 'heartbeat':  return norm(x + heartbeat_track(N, sr))
    if name == 'walkie':        # 军用步话机/头盔: 带通+过载+静电底噪
        v = softclip(bandpass(x, sr, 350, 3200), drive=3.0, out=0.75)
        noise = bandpass(rng.standard_normal(len(x)), sr, 1000, 4000)
        return v + 0.03 * noise
    if name == 'cathedral':     # 大教堂: 长混响+空气感
        return reverb(x, sr, decay_s=4.5, wet=0.45)
    if name == 'vinyl':         # 黑胶电台: 高切+中频凸+颤音+底噪
        v = lowpass(x, sr, 7000)
        b, a = signal.butter(2, [800/(sr/2), 1200/(sr/2)], btype='band')
        v = v + 0.3 * signal.lfilter(b, a, v)
        v = wow_flutter(v, sr)
        noise = lowpass(rng.standard_normal(len(v)), sr, 2000)
        return v + 0.03 * noise
    if name == 'helmet':        # 宇航员头盔通讯: 带通+轻过载
        return softclip(bandpass(x, sr, 400, 3000), drive=2.0, out=0.85)
    if name == 'robot':         # 失控AI: 合唱+flanger金属感
        v = modulated_delay(x, sr, base_ms=18, depth_ms=4, rate_hz=1.2, mix=0.5)
        return modulated_delay(v, sr, base_ms=6, depth_ms=2, rate_hz=0.5, mix=0.6, feedback=0.3)
    raise SystemExit(f'未知效果: {name}')

if __name__ == '__main__':
    src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
    sr, x = read_wav(src)
    write_wav(dst, sr, fx(name, x, sr))
    print(f'{name}: ok')

EFFECTS = ['room','corridor','cockpit','close','far','submarine','temple','phone','radio40','gramophone','megaphone','underwater','wall','tv','cassette','kid','giant','ringmod','chipmunk','demon','harmony','vibrato','tremolo','phaser','pingpong','autowah','radioloud','reverse','slow','bit8','binaural','heartbeat','walkie','cathedral','vinyl','helmet','robot']
