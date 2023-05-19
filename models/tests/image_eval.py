import torch
import onnxruntime
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from openpilot_torch import OpenPilotModel
import os
import csv
import pandas as pd

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def estimate_torch(model, input_imgs, desire, traffic_convention, recurrent_state):
    if not isinstance(input_imgs, torch.Tensor):
        input_imgs = torch.tensor(input_imgs).float()
        desire = torch.tensor(desire).float()
        traffic_convention = torch.tensor(traffic_convention).float()
        recurrent_state = torch.tensor(recurrent_state).float()
    
    out = model(input_imgs, desire, traffic_convention, recurrent_state).detach().numpy()

    lead = out[0, 5755:6010]
    drel = []
    for t in range(6):
        x_predt = lead[4*t::51]
        if t < 3:
            prob = lead[48+t::51]
            current_most_likely_hypo = np.argmax(prob)
            drelt = x_predt[current_most_likely_hypo]
        else:
            drelt = np.mean(x_predt)
        drel.append(drelt)

    rec_state = out[:, -512:]
    return drel, rec_state

def estimate_onnx(model, input_imgs, desire, traffic_convention, recurrent_state):
    out = model.run(['outputs'], {
        'input_imgs': input_imgs,
        'desire': desire,
        'traffic_convention': traffic_convention,
        'initial_state': recurrent_state
    })

    out = np.array(out[0])

    lead = out[0, 5755:6010]
    drel = []
    for t in range(6):
        x_predt = lead[4*t::51]
        if t < 3:
            prob = lead[48+t::51]
            current_most_likely_hypo = np.argmax(prob)
            drelt = x_predt[current_most_likely_hypo]
        else:
            drelt = np.mean(x_predt)
        drel.append(drelt)

    rec_state = out[:, -512:]
    return drel, rec_state

def parse_image(frame):
	H = (frame.shape[0]*2)//3
	W = frame.shape[1]
	parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

	parsed[0] = frame[0:H:2, 0::2] # even column, even row
	parsed[1] = frame[1:H:2, 0::2] # odd column, even row
	parsed[2] = frame[0:H:2, 1::2] # even column, odd row
	parsed[3] = frame[1:H:2, 1::2] # odd column, even row
	parsed[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
	parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))

	return parsed

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 256)) # resize to fit model requirements
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420) # change color channel to YUV rather than BGR
    img = parse_image(img) # parse according to model input

    return img


def prepare_input(img1_path, img2_path, traffic_convention='right'):
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    input_imgs = np.r_[img1, img2][np.newaxis, ...]

    desire = np.zeros(shape=(1, 8))
    desire[0, 0] = 1
    if traffic_convention == 'left':
        traffic_convention = np.array([[0, 1]])
    else:
        traffic_convention = np.array([[1, 0]])
    recurrent_state = np.zeros(shape=(1, 512))

    return input_imgs.astype(np.float32), desire.astype(np.float32), traffic_convention.astype(np.float32), recurrent_state.astype(np.float32)

def load_torch_model():
    model = OpenPilotModel()
    model.load_state_dict(torch.load(r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_torch_weights.pth"))
    model.eval()

    return model

def run_comparison(img1_path, img2_path, traffic_convention='right', verbose=True, rec_states=None):
    input_imgs, desire, traffic_convention, recurrent_state0 = prepare_input(img1_path, img2_path, traffic_convention)
    
    if rec_states is None:
        torch_rec_state = recurrent_state0
        onnx_rec_state = recurrent_state0
    else:
        torch_rec_state, onnx_rec_state = rec_states


    torch_model = load_torch_model()
    torch_drel, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)

    if verbose:
        print('######## PyTorch Output #########')
        print('Lead vehicle relative distance (0, 2, 4, 6, 8, 10) s:', torch_drel)
        print()

    onnx_model = onnxruntime.InferenceSession(r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_server3.onnx")
    onnx_drel, onnx_rec_state = estimate_onnx(onnx_model, input_imgs, desire, traffic_convention, onnx_rec_state)
    
    if verbose:
        print('######## ONNX Output #########')
        print('Lead vehicle relative distance (0, 2, 4, 6, 8, 10) s:', onnx_drel)
        print()

    drel_rmse = np.sqrt(mean_squared_error(torch_drel, onnx_drel))

    if verbose:
        print('dRel RMSE:', drel_rmse)

    rec_states = (torch_rec_state, onnx_rec_state)
    return torch_drel, onnx_drel, drel_rmse, rec_states

# analyze all images
def run_comparison_all():
    with open('../results/golden-left-75m/image_eval_new.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            row = ['Image',
                   'Torch_dRel0', 'Torch_dRel2', 'Torch_dRel4', 'Torch_dRel6', 'Torch_dRel8', 'Torch_dRel10',
                   'ONNX_dRel0', 'ONNX_dRel2', 'ONNX_dRel4', 'ONNX_dRel6', 'ONNX_dRel8', 'ONNX_dRel10', 
                   'RMSE']
            writer.writerow(row)
        

    dataset_path = r"E:\golden-left-75m\golden-left-75m\imgs"
    paths = [p for p in os.listdir(dataset_path) if p.endswith('.png')]

    rec_states = None
    for i in range(1, len(paths)):
        img1_path = os.path.join(dataset_path, paths[i-1])
        img2_path = os.path.join(dataset_path, paths[i])

        torch_d_rel, onnx_d_rel, drel_rmse, rec_states = run_comparison(img1_path, img2_path, traffic_convention='right', verbose=False, rec_states=rec_states)
        print(i, torch_d_rel[0])
        # print('\t', rec_states[0].shape)
        with open('../results/golden-left-75m/image_eval_new.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            row = [i] + list(torch_d_rel) + list(onnx_d_rel) + [drel_rmse]
            writer.writerow(row)
    
    print('Done!')

def collect_results():
    results_path = '../results/golden-left-75m/image_eval_new.csv'
    df = pd.read_csv(results_path)

    for t in range(0, 11, 2):
        error = df['Torch_dRel'+str(t)] - df['ONNX_dRel'+str(t)]
        rmse = np.sqrt(np.mean(error.pow(2)))
        print('dRel RMSE for time {}: {}'.format(t, rmse))

    rmse = np.mean(df['RMSE'])
    print('Overall RMSE:', rmse)

if __name__ == '__main__':
    # img_id = 1
    # img1_path = f"E:\\golden-left-75m\\golden-left-75m\\imgs\\{img_id}.png"
    # img2_path = f"E:\\golden-left-75m\\golden-left-75m\\imgs\\{img_id+1}.png"
    # img1_path = '1.png'
    # img2_path = '2.png'
    # run_comparison(img1_path, img2_path, verbose=True)
    
    # run_comparison_all()
    collect_results()