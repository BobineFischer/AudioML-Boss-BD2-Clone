# Boss BD-2 Emulator using Deep Learning (PyTorch)

**Modélisation d'une pédale d'effet analogique par Deep Learning**

This project explores the use of Recurrent Neural Networks (LSTM) to clone the nonlinear distortion and harmonic characteristics of a physical analog guitar pedal (Boss BD-2 Blues Driver). 

## Audio Demos

You can listen to the high-quality audio comparisons in the shared Google Drive folder below:

🔗 **[Click here to listen to the Audio Demos on Google Drive](https://drive.google.com/drive/folders/1m71JuJhoZnlrqUgSNGGDVgVfYDNnQnot?usp=sharing)**

**Files included in the drive:**
* **`DI.wav`** : Input (Clean DI, used as the raw dataset)
* **`BD.wav`** : Target (Real Boss BD-2, used as the target dataset)
* **`test_di.wav`** : Unseen clean audio for testing
* **`ai_bd2_output.wav`** : Final Output (AI Generated BD-2 Distortion)

---

## Tech Stack

* **Framework:** PyTorch (with Apple Silicon MPS Acceleration)
* **Audio Processing:** Soundfile, Numpy
* **Architecture:** LSTM (Long Short-Term Memory) network tailored for 1D audio time-series.

## Key Technical Challenges Solved

### 1. Sample-Accurate Time Alignment (RTL Compensation)
To train the LSTM effectively, the physical Round-Trip Latency (RTL) caused by the Audio Interface (D/A and A/D conversion) was manually measured and compensated at the waveform level to prevent phase cancellation and ensure perfect input-target matching.

### 2. Zipper Noise Elimination via Overlap-Add (OLA)
Inference on continuous audio tracks initially produced steep discontinuities (click/pop artifacts) at chunk boundaries. This was resolved by implementing a standard DSP **Overlap-Add (OLA)** method with a 50% overlapping **Hann Window** to ensure seamless crossfading between predicted chunks.

### 3. GPU Memory Management (OOM Prevention)
To handle long audio files (e.g., full songs) without exceeding the unified memory limits of the Apple M2 chip, a mini-batching mechanism was implemented during the inference phase, ensuring stable and scalable audio rendering.
