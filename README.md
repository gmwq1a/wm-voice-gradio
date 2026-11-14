# 万事屋云端滤镜 · Gradio 版（HF Spaces 备用存档）

voice_dsp 37 种语音音效的 Gradio 网页版。2026-11-14 写好并本地实测通过（apply() 出活、5 分钟上限正确报错），HF Spaces 线当日取消待命，代码存档备用。

## 文件
- `app.py` — Gradio 界面（37 效果分类下拉、上传/录音输入、wav 输出、5 分钟上限）
- `voice_dsp.py` — 37 效果 DSP 库（numpy+scipy，11-12 全部人工验收；含分块混响，长音频内存安全）
- `requirements.txt` — gradio / numpy / scipy

## 发布到 HF Spaces（将来要做时）
1. 建 Space：sdk=gradio，硬件 CPU basic（免费）。注意免费账号需满 30 天才能建 Space。
2. 把本仓库三个文件传上去（或 HF 端 `git clone` 本 repo 后 push 到 Space 的 git）。
3. 无需额外环境变量。

线上同款：Render 版 API 在 wm-tools（/api/audio/jobs），契约见字幕猫线文档。
