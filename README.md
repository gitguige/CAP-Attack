# CAP-Attack
This repository contains source code to demonstrate the effectiveness of CAP-Attack, a Context-Aware Perception Attack that stealthily undermines DNN-based Adaptive Cruise Control (ACC) systems. This work was propsed in the paper *"Runtime Stealthy Perception Attacks against DNN-based Adaptive Cruise Control Systems"* by Xugui Zhou et al. CAP-Attack uses a novel optimization-based approach and adaptive algorithm to generate an adversarial patch for input camera frames at runtime. The addition of the adversarial patch to input frames fools the DNN model in the ACC system and causes unsafe acceleration to occur before detection and migitation by the driver or safety mechanisms. The code contained here performs the attack offline on precollected dashcam videos or images to illustrate adversarial patch generation and application.


HOT TO CITE:

X. Zhou, A. Chen, Ma. Kouzel, H. Ren, M. McCarty, C. Nita-Rotaru, and H. Alemzadeh, "Runtime Stealthy Perception Attacks against DNN-based Adaptive Cruise Control Systems," in ACM Asia Conference on Computer and Communications Security (ASIA CCS ’25), 2025.

```
@inproceedings{zhou2025runtime,
  title={Runtime Stealthy Perception Attacks against DNN-based Adaptive Cruise Control Systems},
  author={Zhou, Xugui and Chen, Anqi and Kouzel, Maxfield and Ren, Haotian and McCarty, Morgan and Nita-Rotaru, Cristina and Alemzadeh, Homa},
  booktitle={20th ACM ASIA Conference on Computer and Communications Security (AsiaCCS)},
  year={2025}
}
```

## Requirements
See `requirements.txt`, or:
- Python 3.9.6
- ONNX 1.14.0
- onnxruntime 1.14.1
- PyTorch 2.0.1
- Ultralytics 8.0.105
- SciKit Learn 1.2.2

## Installation
- Clone this repository
- Install dependencies
    - `cd /your/path/here/CAP-Attack/`
    - `pip install -r requirements.txt`

## Usage
Once environment is set up, run `python video_eval.py [verbose]` in the `./attack_tests/` directory. \
The `verbose` argument is optional and 0 is minimal output, 1 outputs a progress message every 200 frames, and 2 outputs status information every frame. 

- To use video data other than the sample data, either modify `CHUNK_DIRECTORY` in `video_eval.py` or copy your data to the `CAP-Attack/data` directory
    - Sample video data is from the [comma2k19](https://github.com/commaai/comma2k19) dataset, and the bulk video analysis script is designed to process data from that dataset so preprocessing steps may need to be tweaked for data from other sources
- To use image data other than the sample data, modify `IMGS_DIRECTORY` in `image_eval.py`
    - Sample image data was generated using the [CARLA Simulator](https://github.com/carla-simulator/carla)

## Reproduction
Once each video has been processed by evaluation script, average and standard deviation metrics for each adversarial patch type's effect on the model's estimate for lead car relative distance and RMSE metrics bewteen the ONNX model from OpenPilot and the PyTorch model will be generated. The 3 types of patches are the CAP-Attack patch (called "optimized" patch), a baseline adversarial patch created using the Fast Gradient Sign Method (FGSM), and a random patch. The deviation reported is the difference between the original ONNX model's prediction for the lead car's current relative distance (i.e. time horizon of 0) and the ONNX model's current relative distance prediction after the patch has been applied to the input images. A higher deviation means the adversarial patch was more effective. The RMSE metric compares how closely the PyTorch version of the model replicates the ONNX version, so a smaller RMSE is desirable as the PyTorch model is used to optimize the patch.

The metrics for the sample video data provided are shown below as an example:
```
### Optimized Patch ###
Overall Deviation (m): 11.651084180493696 (avg), 22.63484539593186 (std)
Deviation by lead vehicle distance (m):
	[0, 20] : 38.077363943039266 (avg), 36.5776798438628, (std)
	[20, 40] : 8.90427834476004 (avg), 15.494763443442293, (std)
	[40, 60] : 8.387780854389723 (avg), 10.33718476335239, (std)
	[60, 80] : 8.560524510934393 (avg), 7.54147929244774, (std)
	[80, 999] : 2.9439739990198 (avg), 4.462562408293018, (std)
RMSE between Torch and ONNX models (overall): 6.267276713996245e-05
RMSE between Torch and ONNX models by lead vehicle distance (m):
	[0, 20] : 3.444669775334201e-06
	[20, 40] : 1.0093929693984696e-05
	[40, 60] : 1.4237172101399019e-05
	[60, 80] : 1.9678622072377293e-05
	[80, 999] : 5.6781725346623996e-05

### Random Patch ###
Overall Deviation (m): 0.13332046169117656 (avg), 2.371943395012448 (std)
Deviation by lead vehicle distance (m):
	[0, 20] : 0.27180057363590004 (avg), 0.7007840254954648, (std)
	[20, 40] : 0.00575331145935357 (avg), 1.1948013612539026, (std)
	[40, 60] : 0.024331187794432645 (avg), 3.6437793782895658, (std)
	[60, 80] : 0.11969888071570547 (avg), 2.2539416298471244, (std)
	[80, 999] : 0.1269166581062539 (avg), 2.6609427543599202, (std)
RMSE between Torch and ONNX models (overall): 8.577053003752715e-05
RMSE between Torch and ONNX models by lead vehicle distance (m):
	[0, 20] : 1.5550596991929194e-05
	[20, 40] : 2.5682208768365973e-05
	[40, 60] : 3.455029405728363e-05
	[60, 80] : 3.8777220009518055e-05
	[80, 999] : 6.130016845637277e-05

### FGSM Patch ###
Overall Deviation (m): 5.783177634569328 (avg), 12.93259418383814 (std)
Deviation by lead vehicle distance (m):
	[0, 20] : 18.78040350591535 (avg), 22.271215346020853, (std)
	[20, 40] : 4.305470687561215 (avg), 9.907001096854028, (std)
	[40, 60] : 3.914400123126338 (avg), 6.109941352896786, (std)
	[60, 80] : 4.629465332007952 (avg), 4.521511241296334, (std)
	[80, 999] : 1.5383108129778476 (avg), 2.965716815064241, (std)
RMSE between Torch and ONNX models (overall): 7.41199034242355e-05
RMSE between Torch and ONNX models by lead vehicle distance (m):
	[0, 20] : 5.178341554007147e-06
	[20, 40] : 1.4959489277539826e-05
	[40, 60] : 2.1295267145026002e-05
	[60, 80] : 2.5453391773584855e-05
	[80, 999] : 6.435675361697162e-05

### No Patch ###
RMSE between Torch and ONNX models (overall): 8.34061235137384e-05
RMSE between Torch and ONNX models by lead vehicle distance (m):
	[0, 20] : 1.5340457384900803e-05
	[20, 40] : 2.467602975449311e-05
	[40, 60] : 3.406872547595069e-05
	[60, 80] : 3.84207320697482e-05
	[80, 999] : 5.89534681149916e-05
```
