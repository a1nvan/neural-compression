# Neural Image Compression Pipeline (Cheng2020)

A tool for automated neural network image compression based on the **Cheng2020** architecture. The project allows you to massively compress data, calculate the actual bitrate BPP and prepare datasets to analyze the impact of compression on the accuracy of detection models (YOLO) and OCR.

---

## Technology stack
* **Core:** PyTorch & CompressAI
* **Models:** `cheng2020_attn` and `cheng2020_anchor`.

---

## Quick Start

### 1. GitHub Codespaces
Just open the project in Codespaces. The environment is configured automatically via `.devcontainer'.
```bash
pip install -r requirements.txt

python src/compressor.py
```
### Google Colab
```bash
!git clone https://github.com/a1nvan/neural-compression.git

%cd neural-compression
!pip install compressai pandas tqdm
```
#### Add photos to /data/raw
```bash
!python src/compressor.py
```
