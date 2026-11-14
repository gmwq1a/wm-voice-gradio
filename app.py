# 万事屋云端滤镜 · Gradio 版（voice_dsp 37 效果）
# HF Space: sdk=gradio, CPU basic (free)
import tempfile, os
import numpy as np
import gradio as gr
from scipy.io import wavfile
import voice_dsp

LABELS = {
    "环境空间": [("room","小房间"),("corridor","走廊"),("cockpit","驾驶舱"),("close","贴耳近场"),
                 ("far","远景回荡"),("submarine","深海潜艇"),("temple","哥特圣殿"),("cathedral","大教堂")],
    "设备介质": [("phone","老式电话"),("radio40","1940s收音机"),("gramophone","留声机"),("megaphone","扩音喇叭"),
                 ("underwater","水下"),("wall","隔墙听"),("tv","电视喇叭"),("cassette","磁带"),
                 ("walkie","步话机"),("vinyl","黑胶"),("helmet","宇航员头盔")],
    "变声":     [("kid","幼化"),("giant","巨人化"),("ringmod","机器人"),("chipmunk","花栗鼠"),
                 ("demon","恶魔低语"),("harmony","和声分身"),("robot","失控AI")],
    "调制":     [("vibrato","颤音"),("tremolo","震音"),("phaser","相位"),("pingpong","乒乓延迟"),("autowah","自动哇音")],
    "动态创意": [("radioloud","广播响度压限"),("reverse","倒放"),("slow","慢速拉伸"),
                 ("bit8","8-bit游戏机"),("binaural","双耳3D环绕"),("heartbeat","心跳呼吸氛围")],
}
CHOICES = [f"{cat} / {label} ({name})" for cat, items in LABELS.items() for name, label in items]
NAME_OF = {f"{cat} / {label} ({name})": name for cat, items in LABELS.items() for name, label in items}
MAX_S = 300

def apply(audio, choice):
    if audio is None:
        raise gr.Error("先上传一段音频")
    name = NAME_OF[choice]
    sr, data = audio
    x = np.asarray(data, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.max(np.abs(x)) or 1.0
    if data.dtype == np.int16:
        x = x / 32768.0
    else:
        x = x / peak
    dur = len(x) / sr
    if dur > MAX_S:
        raise gr.Error(f"这段音频 {dur:.0f} 秒，超过免费额度 {MAX_S//60} 分钟上限；更长的请拆段处理")
    y = voice_dsp.fx(name, x, sr)
    y = np.clip(np.asarray(y, dtype=np.float64), -1, 1)
    out = (y * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wavfile.write(path, sr, out)
    return path

with gr.Blocks(title="万事屋云端滤镜") as demo:
    gr.Markdown("# 万事屋云端滤镜\n37 种语音音效（环境空间 / 设备介质 / 变声 / 调制 / 动态创意），纯 DSP 实时处理。免费额度单段最长 5 分钟。")
    with gr.Row():
        inp = gr.Audio(sources=["upload", "microphone"], type="numpy", label="输入音频")
        eff = gr.Dropdown(choices=CHOICES, value=CHOICES[0], label="滤镜")
    btn = gr.Button("一键出活", variant="primary")
    out = gr.Audio(label="成品（可下载）", type="filepath")
    btn.click(apply, [inp, eff], out)
    gr.Markdown("变调类（幼化/巨人化/花栗鼠/恶魔/慢速/倒放）时长会变，属磁带变速原理；双耳3D与乒乓延迟输出立体声。")

demo.launch()
