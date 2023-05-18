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

    lead_prob = sigmoid(out[:, 6010:6013])
    x_rel = out[:, 5755:5779:4]
    y_rel = out[:, 5755+1:5779:4]
    d_rel = np.sqrt(np.power(x_rel, 2) + np.power(y_rel, 2))

    speed = out[:, 5755+2:5779:4]

    return lead_prob, d_rel, speed

def estimate_onnx(model, input_imgs, desire, traffic_convention, recurrent_state):
    out = model.run(['outputs'], {
        'input_imgs': input_imgs,
        'desire': desire,
        'traffic_convention': traffic_convention,
        'initial_state': recurrent_state
    })

    out = np.array(out[0])
    print(out[:, 5755:])

    lead_prob = sigmoid(out[:, 6010:6013])
    x_rel = out[:, 5755:5779:4]
    y_rel = out[:, 5755+1:5779:4]
    d_rel = np.sqrt(np.power(x_rel, 2) + np.power(y_rel, 2))

    speed = out[:, 5755+2:5779:4]

    return lead_prob, d_rel, speed

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

def run_comparison(img1_path, img2_path, traffic_convention='right', verbose=True):
    input_imgs, desire, traffic_convention, recurrent_state = prepare_input(img1_path, img2_path, traffic_convention)


    torch_model = load_torch_model()
    torch_lead_prob, torch_d_rel, torch_speed = estimate_torch(torch_model, input_imgs, desire, traffic_convention, recurrent_state)

    if verbose:
        print('######## PyTorch Output #########')
        print('Lead vehicle prob:', torch_lead_prob)
        print('Lead vehicle relative distance (0, 2, 4, 6, 8, 10) s:', torch_d_rel)
        print('Lead vehicle speed (0, 2, 4, 6, 8, 10) s:', torch_speed)
        print()

    onnx_model = onnxruntime.InferenceSession(r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_server3.onnx")
    onnx_lead_prob, onnx_d_rel, onnx_speed = estimate_onnx(onnx_model, input_imgs, desire, traffic_convention, recurrent_state)
    
    if verbose:
        print('######## ONNX Output #########')
        print('Lead vehicle prob:', onnx_lead_prob)
        print('Lead vehicle relative distance (0, 2, 4, 6, 8, 10) s:', onnx_d_rel)
        print('Lead vehicle speed (0, 2, 4, 6, 8, 10) s:', onnx_speed)
        print()

    lead_prob_rmse = np.sqrt(mean_squared_error(torch_lead_prob, onnx_lead_prob))
    drel_rmse = np.sqrt(mean_squared_error(torch_d_rel, onnx_d_rel))
    speed_rmse = np.sqrt(mean_squared_error(torch_speed, onnx_speed))

    if verbose:
        print('Lead Prob RMSE:', lead_prob_rmse)
        print('dRel RMSE:', drel_rmse)
        print('Speed RMSE:', speed_rmse)

    return torch_d_rel, onnx_d_rel, drel_rmse

if __name__ == '__main__':
    img_id = 1
    # img1_path = f"E:\\golden-left-75m\\golden-left-75m\\imgs\\{img_id}.png"
    # img2_path = f"E:\\golden-left-75m\\golden-left-75m\\imgs\\{img_id+1}.png"
    img1_path = '1.png'
    img2_path = '2.png'
    run_comparison(img1_path, img2_path, verbose=True)

    # analyze all images
    # with open('../results/golden-left-75m/image_eval.csv', 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         row = ['Image',
    #                'Torch_dRel0', 'Torch_dRel2', 'Torch_dRel4', 'Torch_dRel6', 'Torch_dRel8', 'Torch_dRel10',
    #                'ONNX_dRel0', 'ONNX_dRel2', 'ONNX_dRel4', 'ONNX_dRel6', 'ONNX_dRel8', 'ONNX_dRel10', 
    #                'RMSE']
    #         writer.writerow(row)
        

    # dataset_path = r"E:\golden-left-75m\golden-left-75m\imgs"
    # paths = [p for p in os.listdir(dataset_path) if p.endswith('.png')]
    # for i in range(1, len(paths)):
    #     print(i)
    #     img1_path = os.path.join(dataset_path, paths[i-1])
    #     img2_path = os.path.join(dataset_path, paths[i])

    #     torch_d_rel, onnx_d_rel, drel_rmse = run_comparison(img1_path, img2_path, traffic_convention='right', verbose=False)

    #     with open('../results/golden-left-75m/image_eval.csv', 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         row = [i] + list(torch_d_rel[0]) + list(onnx_d_rel[0]) + [drel_rmse]
    #         writer.writerow(row)
    

    # collect results
    # results_path = '../results/golden-left-75m/image_eval.csv'
    # df = pd.read_csv(results_path)

    # for t in range(0, 11, 2):
    #     error = df['Torch_dRel'+str(t)] - df['ONNX_dRel'+str(t)]
    #     rmse = np.sqrt(np.mean(error.pow(2)))
    #     print('dRel RMSE for time {}: {}'.format(t, rmse))

    # rmse = np.mean(df['RMSE'])
    # print('Overall RMSE:', rmse)

