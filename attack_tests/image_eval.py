import copy
import torch
import onnxruntime
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from openpilot_torch import OpenPilotModel
import os
import csv
import pandas as pd
from ultralytics import YOLO


# from DRP-attack repo: https://github.com/ASGuard-UCI/DRP-attack
class AdamOptTorch:

    def __init__(self, size, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=torch.float32):
        self.exp_avg = torch.zeros(size, dtype=dtype)
        self.exp_avg_sq = torch.zeros(size, dtype=dtype)
        self.beta1 = torch.tensor(beta1)
        self.beta2 = torch.tensor(beta2)
        self.eps = eps
        self.lr = lr
        self.step = 0

    def update(self, grad):

        self.step += 1

        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step

        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)

        denom = (torch.sqrt(self.exp_avg_sq) / torch.sqrt(bias_correction2)) + self.eps

        step_size = self.lr / bias_correction1

        return step_size / denom * self.exp_avg

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def estimate_torch(model, input_imgs, desire, traffic_convention, recurrent_state):
    if not isinstance(input_imgs, torch.Tensor):
        input_imgs = torch.tensor(input_imgs).float()
    if not isinstance(desire, torch.Tensor):
        desire = torch.tensor(desire).float()
        traffic_convention = torch.tensor(traffic_convention).float()
    if not isinstance(recurrent_state, torch.Tensor): 
        recurrent_state = torch.tensor(recurrent_state).float()
    
    out = model(input_imgs, desire, traffic_convention, recurrent_state)

    lead = out[0, 5755:6010]
    drel = []
    for t in range(6):
        x_predt = lead[4*t::51]
        if t < 3:
            prob = lead[48+t::51]
            current_most_likely_hypo = torch.argmax(prob)
            drelt = x_predt[current_most_likely_hypo]
        else:
            drelt = torch.mean(x_predt)
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

def load_torch_model(path):
    model = OpenPilotModel()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

def run_comparison(img1_path, img2_path, models, traffic_convention='right', verbose=True, rec_states=None):
    torch_model_path, onnx_model_path = models
    torch_model = load_torch_model(torch_model_path)
    onnx_model = onnxruntime.InferenceSession(onnx_model_path)

    input_imgs, desire, traffic_convention, recurrent_state0 = prepare_input(img1_path, img2_path, traffic_convention)
    
    if rec_states is None:
        torch_rec_state = recurrent_state0
        onnx_rec_state = recurrent_state0
    else:
        torch_rec_state, onnx_rec_state = rec_states


    # torch_model = load_torch_model()
    torch_drel, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
    torch_drel = [v.detach().numpy() for v in torch_drel]
    print(torch_drel[0])
    
    if verbose:
        print('######## PyTorch Output #########')
        print('Lead vehicle relative distance (0, 2, 4, 6, 8, 10) s:', torch_drel)
        print()

    # onnx_model = onnxruntime.InferenceSession(r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_server3.onnx")
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

def build_yuv_patch(thres, patch_dim, patch_start):
    patch_height, patch_width = patch_dim
    patch_y, patch_x = patch_start # top-left is (0, 0) horizontal is x, vertical is y
    h_ratio = 256/874
    w_ratio = 512/1164

    y_patch_height = int(patch_height*h_ratio)
    y_patch_width = int(patch_width*w_ratio)
    y_patch_h_start = int(patch_y*h_ratio)
    y_patch_w_start = int(patch_x*w_ratio)

    y_patch = thres * np.random.rand(y_patch_height, y_patch_width).astype('float32')
    # u_patch = thres * np.random.rand()
    h_bounds = (y_patch_h_start, y_patch_h_start+y_patch_height)
    w_bounds = (y_patch_w_start, y_patch_w_start+y_patch_width)
    return y_patch, h_bounds, w_bounds

def parse_image_multi(frames):
    n = frames.shape[0] # should always be 2
    H = (frames.shape[1]*2)//3 # 128
    W = frames.shape[2] # 256

    parsed = torch.zeros(size=(n, 6, H//2, W//2)).float()
    

    parsed[:, 0] = frames[:, 0:H:2, 0::2] # even column, even row
    parsed[:, 1] = frames[:, 1:H:2, 0::2] # odd column, even row
    parsed[:, 2] = frames[:, 0:H:2, 1::2] # even column, odd row
    parsed[:, 3] = frames[:, 1:H:2, 1::2] # odd column, even row
    parsed[:, 4] = frames[:, H:H+H//4].reshape((n, H//2,W//2))
    parsed[:, 5] = frames[:, H+H//4:H+H//2].reshape((n, H//2,W//2))

    return parsed

def run_patched_comparison(img1_path, img2_path, models, traffic_convention='right', verbose=True, rec_states=None):
    pred_times = [0, 2, 4, 6, 8, 10]
    dim = (512, 256)
    thres = 1
    mask_iterations = 10

    torch_model_path, onnx_model_path = models
    torch_model = load_torch_model(torch_model_path)
    onnx_model = onnxruntime.InferenceSession(onnx_model_path)
    yolo_model = YOLO("yolov8n.pt")

    # set up inputs
    desire = np.zeros(shape=(1, 8)).astype('float32')
    traffic_convention = np.array([[1, 0]]).astype('float32')
    if rec_states is None:
        torch_rec_state = np.zeros(shape=(1, 512)).astype('float32')
        onnx_rec_state = np.zeros(shape=(1, 512)).astype('float32')
    else:
        torch_rec_state, onnx_rec_state = rec_states

    
    # read in images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    ### locate and build patch for image 2 ###
    results = yolo_model(img2, verbose=False)
    boxes = results[0].boxes

    # choose largest box found
    max_size = 0
    for i in range(len(boxes)):
        box_ = boxes[i].xyxy
        size = (box_[0, 3] - box_[0, 1])*(box_[0, 2] - box_[0, 0])
        if size > max_size:
            box = box_
            max_size = size

    # if no object found, attack 1 pixel in the center of the image
    if len(boxes) == 0:
        box = torch.tensor([[437, 582, 438, 583]])
    
    if isinstance(box, torch.Tensor):
        box = box.int().numpy()
    else:
        box = box.astype('int32')

    patch_start = (box[0, 1], box[0, 0])
    patch_dim = (box[0, 3] - box[0, 1], box[0, 2] - box[0, 0])
    
    patch, h_bounds, w_bounds = build_yuv_patch(thres, patch_dim, patch_start)
    patch = torch.tensor(patch, requires_grad=True)
    adam = AdamOptTorch(patch.shape, lr = 1)

    # convert images to YUV I420
    img1 = cv2.resize(img1, dim)
    img2 = cv2.resize(img2, dim)
    img1_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV_I420).astype('float32')
    img2_yuv = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV_I420).astype('float32')
    proc_images = [img1_yuv, img2_yuv]

    # initial pass through
    input_imgs = parse_image_multi(torch.tensor(np.array(proc_images))).reshape(1, 12, 128, 256)
    torch_init_drel, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
    onnx_init_drel, onnx_rec_state = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)

    torch_init_drel = [v.detach().numpy() for v in torch_init_drel]
    init_drel_rmse = np.sqrt(mean_squared_error(torch_init_drel, onnx_init_drel))
    init_results = (torch_init_drel, onnx_init_drel, init_drel_rmse)
    print(onnx_init_drel[0], torch_init_drel[0])

    ### optimize patch ###
    for it in range(mask_iterations):
        # print(it)
        # apply new patch to the image
        patch = torch.clip(patch, -thres, thres)
        tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
        tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += patch

        tmp_pimgs = parse_image_multi(tmp_pimgs)
        input_imgs = tmp_pimgs.reshape(1, 12, 128, 256)

        # pass newly masked image through the model
        torch_drel, _ = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)

        if it == 0:
            print(torch_drel[0])
        
        patch.retain_grad()
        torch_drel[0].backward(retain_graph=True)
        grad = patch.grad
        
        # update patch
        update = adam.update(grad)
        patch = patch.clone().detach().requires_grad_(True) + update
        torch_rec_state = torch_rec_state.clone().detach()

    # create input with optimized patch
    patch = torch.clip(patch, -thres, thres)
    tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
    tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += patch
    tmp_pimgs = parse_image_multi(tmp_pimgs)
    input_imgs = tmp_pimgs.reshape(1, 12, 128, 256)

    ### send patched input through model ###
    torch_drel, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
    torch_drel = [v.detach().numpy() for v in torch_drel]
    onnx_drel, onnx_rec_state = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)

    if verbose:
        print('######## PyTorch Output #########')
        print('Lead vehicle relative distance (0, 2, 4, 6, 8, 10) s:', torch_drel)
        print()
    
    if verbose:
        print('######## ONNX Output #########')
        print('Lead vehicle relative distance (0, 2, 4, 6, 8, 10) s:', onnx_drel)
        print()

    drel_rmse = np.sqrt(mean_squared_error(torch_drel, onnx_drel))

    if verbose:
        print('dRel RMSE:', drel_rmse)

    print(torch_drel[0], '| Patch size:', patch_dim)

    rec_states = (torch_rec_state.clone().detach(), onnx_rec_state)
    optmask_results = (torch_drel, onnx_drel, drel_rmse)
    return init_results,  optmask_results, rec_states

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

    # torch_model = load_torch_model()
    # onnx_model = onnxruntime.InferenceSession(r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_server3.onnx")

    rec_states = None
    for i in range(1, len(paths)):
        img1_path = os.path.join(dataset_path, paths[i-1])
        img2_path = os.path.join(dataset_path, paths[i])

        torch_drel, onnx_drel, drel_rmse, rec_states = run_comparison(img1_path, img2_path, traffic_convention='right', verbose=False, rec_states=rec_states)
        print(i, torch_drel[0])
        # print('\t', rec_states[0].shape)
        with open('../results/golden-left-75m/image_eval_new.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            row = [i] + list(torch_drel) + list(onnx_drel) + [drel_rmse]
            writer.writerow(row)
    
    print('Done!')


# analyze all images
def run_patched_comparison_all(system='linux'):
    if system == 'linux':
        result_path = '/home/student/Max/results/CARLA_images/image_eval.csv'
        torch_model_path = '/home/student/Max/supercombo_torch_weights.pth'
        onnx_model_path = '/home/student/supercombo_inf/supercombo_server3.onnx'
        dataset_path = '/home/student/Max/golden-left-75m/imgs'
    else:
        result_path = '../results/golden-left-75m/image_eval_new.csv'
        torch_model_path = r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_torch_weights.pth"
        onnx_model_path = r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_server3.onnx"
        # dataset_path = r"E:\golden-left-75m\golden-left-75m\imgs"
        dataset_path = '../dataset/golden-left-75m/imgs'

    with open(result_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = ['Image',
                   'Torch_init_dRel0', 'Torch_init_dRel2', 'Torch_init_dRel4', 'Torch_init_dRel6', 'Torch_init_dRel8', 'Torch_init_dRel10',
                   'Torch_optmask_dRel0', 'Torch_optmask_dRel2', 'Torch_optmask_dRel4', 'Torch_optmask_dRel6', 'Torch_optmask_dRel8', 'Torch_optmask_dRel10',
                   'ONNX_init_dRel0', 'ONNX_init_dRel2', 'ONNX_init_dRel4', 'ONNX_init_dRel6', 'ONNX_init_dRel8', 'ONNX_init_dRel10', 
                   'ONNX_optmask_dRel0', 'ONNX_optmask_dRel2', 'ONNX_optmask_dRel4', 'ONNX_optmask_dRel6', 'ONNX_optmask_dRel8', 'ONNX_optmask_dRel10',
                   'RMSE_init', 'RMSE_optmask']
            writer.writerow(row)
        

    # torch_model = load_torch_model(torch_model_path)
    # onnx_model = onnxruntime.InferenceSession(onnx_model_path)

    paths = [p for p in os.listdir(dataset_path) if p.endswith('.png')]

    rec_states = None
    for i in range(1, len(paths)):
        img1_path = os.path.join(dataset_path, paths[i-1])
        img2_path = os.path.join(dataset_path, paths[i])

        
        init_results,  optmask_results, rec_states = run_patched_comparison(
            img1_path, 
            img2_path, 
            models=(torch_model_path, onnx_model_path),
            traffic_convention='right', 
            verbose=False, 
            rec_states=rec_states,
        )

        torch_init_drel, onnx_init_drel, init_drel_rmse = init_results
        torch_optmask_drel, onnx_optmask_drel, opt_drel_rmse = optmask_results
        print(i)
        with open(result_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [i] + list(torch_init_drel) + list(torch_optmask_drel) + list(onnx_init_drel) + list(onnx_optmask_drel) + [init_drel_rmse, opt_drel_rmse]
            writer.writerow(row)
    
    print('Done!')


def collect_results():
    results_path = '../results/golden-left-75m/image_eval_new.csv'
    df = pd.read_csv(results_path)

    for t in range(0, 11, 2):
        error = df['Torch_dRel'+str(t)] - df['ONNX_dRel'+str(t)]
        rmse = np.sqrt(np.mean(error.pow(2)))
        print('dRel RMSE for time {}: {}'.format(t, rmse))


    bounds = [0, 20, 40, 60, 80, 2147483647]
    t_cols = ['Torch_dRel'+str(t) for t in range(0, 12, 2)]
    o_cols = ['ONNX_dRel'+str(t) for t in range(0, 12, 2)]

    for i in range(1, len(bounds)):
        t_predictions_within_bounds = np.all([df[t_cols] > bounds[i-1], df[t_cols] <= bounds[i]], axis=0) # all entries where estimate is within (L, U]
        o_predictions_within_bounds = np.all([df[o_cols] > bounds[i-1], df[o_cols] <= bounds[i]], axis=0)
        if not np.all(t_predictions_within_bounds == o_predictions_within_bounds):
            print(':( at ', t)
        t_mask = np.array(df[t_cols]*t_predictions_within_bounds)
        o_mask = np.array(df[o_cols]*o_predictions_within_bounds)
        rmse = np.sqrt(np.mean(np.power(o_mask - t_mask, 2)))
        print('dRel RMSE for estimates within ({}, {}]'.format(bounds[i-1], bounds[i]), rmse)

    rmse = np.mean(df['RMSE'])
    print('Overall RMSE:', rmse)

def collect_results_patched(col_type='init', system='linux'):
    if system == 'linux':
        results_path = '/home/student/Max/results/CARLA_images/image_eval.csv'
    else:
        results_path = '../results/golden-left-75m/image_eval_new.csv'

    df = pd.read_csv(results_path)

    for t in range(0, 11, 2):
        error = df[f'Torch_{col_type}_dRel{t}'] - df[f'ONNX_{col_type}_dRel{t}']
        rmse = np.sqrt(np.mean(error.pow(2)))
        print('dRel RMSE for time {}: {}'.format(t, rmse))


    bounds = [0, 20, 40, 60, 80, 2147483647]
    t_cols = [f'Torch_{col_type}_dRel{t}' for t in range(0, 12, 2)]
    o_cols = [f'ONNX_{col_type}_dRel{t}' for t in range(0, 12, 2)]

    for i in range(1, len(bounds)):
        # t_predictions_within_bounds = np.all([df[t_cols] > bounds[i-1], df[t_cols] <= bounds[i]], axis=0) # all entries where estimate is within (L, U]
        o_predictions_within_bounds = np.all([df[o_cols] > bounds[i-1], df[o_cols] <= bounds[i]], axis=0)
        # if not np.all(t_predictions_within_bounds == o_predictions_within_bounds):
        #     print(':( at ', t)
        t_mask = np.array(df[t_cols]*o_predictions_within_bounds)
        o_mask = np.array(df[o_cols]*o_predictions_within_bounds)
        rmse = np.sqrt(np.mean(np.power(o_mask - t_mask, 2)))
        print('dRel RMSE for estimates within ({}, {}] :'.format(bounds[i-1], bounds[i]), rmse)

    rmse = np.mean(df[f'RMSE_{col_type}'])
    print('Overall RMSE:', rmse)

if __name__ == '__main__':
    # img_id = 1
    # img1_path = f"E:\\golden-left-75m\\golden-left-75m\\imgs\\{img_id}.png"
    # img2_path = f"E:\\golden-left-75m\\golden-left-75m\\imgs\\{img_id+1}.png"
    # img1_path = '1.png'
    # img2_path = '2.png'
    # run_comparison(img1_path, img2_path, verbose=True)
    
    system='linux'
    # print('Using ' + system + ' file names')
    # run_patched_comparison_all(system)
    print('### init results ###')
    collect_results_patched('init', system)
    print('### optmask results ###')
    collect_results_patched('optmask', system)
    
