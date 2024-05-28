# CAP-Attack
Context-Aware Perception Attack

This repository contains source code to demonstrate the effectiveness of CAP-Attack, a context-aware perception attack that stealthily undermines DNN based Adaptive Cruise Control (ACC) systems. This work was poropsed in the paper *"Runtime Stealthy Perception Attacks against DNN-based Adaptive Cruise Control Systems"* by Xugui Zhou*, Anqi Chen&dagger;, Maxfield Kouzel*, Hoatian Ren*, Morgan McCarty&dagger;, Christina Nita-Rotaru&dagger;, and Homa Alemzadeh*. CAP-Attack uses a novel optimization-based approach and adaptive algorithm to generate an adversarial patch to input camera frames at runtime in order to fool the DNN model in the ACC system and cause unsafe acceleration being detection or migitation by the driver or safety mechanisms. The code contained here performs the attack offline on precollected dashcam videos or images to illustrate adversarial patch generation.

\* *University of Virginia* \
&dagger; *Northeastern University*

## Requirements
See `requirements.txt`, or:
- Python 3.9.6
- ONNX 1.14.0
- onnxruntime 1.14.1
- PyTorch 2.0.1
- Ultralytics 8.0.105 (for YOLO)
- SciKit Learn 1.2.2

## Installation
- Clone this repository
- Install dependencies
    - `cd /your/path/here/CAP-Attack/`
    - `pip install -r requirements.txt`
- To use data other than the sample data, either modify `CHUNK_DIRECTORY` in `video_eval.py` or copy your data to the `CAP-Attack/data` directory

## Usage
Once environment is set up, run `python video_eval.py [verbose]`. \
The `verbose` argument is optional and 0 is minimal output, 1 outputs every 200 frames, and 2 outputs information every frame.

## Reproduction


## Acknowledgement
