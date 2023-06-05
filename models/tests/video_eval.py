import torch
import onnxruntime
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from openpilot_torch import OpenPilotModel
import pandas as pd
import os
import sys
from datetime import datetime
from ultralytics import YOLO
import copy

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

def load_torch_model():
    model = OpenPilotModel()
    model.load_state_dict(torch.load('/home/student/Max/supercombo_torch_weights.pth'))
    model.eval()

    return model

def dist(x, y):
    return torch.cdist(x, y, p=2.0)


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

def analyze_video(video_path, video_id):

    onnx_model = onnxruntime.InferenceSession('/home/student/supercombo_inf/supercombo_server3.onnx')
    # onnx_model = onnxruntime.InferenceSession('/home/student/Max/supercombo_alt_version.onnx')
    torch_model = load_torch_model()
    yolo_model = YOLO("yolov8n.pt")
    
	
    cap = cv2.VideoCapture(video_path)
    # 2nd input img for 1st frame is 0's (directly replicates OpenPilot)
    proc_images = [np.zeros(shape=(384, 512))]
    parsed_images = [np.zeros(shape=(6, 128, 256))]

    width = 512
    height = 256
    dim = (width, height)

    # initial values for non-image inputs
    desire = np.zeros(shape=(1, 8)).astype('float32')
    traffic_convention = np.array([[1, 0]]).astype('float32')
    torch_rec_state = np.zeros(shape=(1, 512)).astype('float32')
    onnx_rec_state = np.zeros(shape=(1, 512)).astype('float32')

    pred_times = [0, 2, 4, 6, 8, 10]
    # PyTorch dRel history (before patch is applied)
    torch_init_drel_hist = {
         'dRel0': [],
         'dRel2': [],
         'dRel4': [],
         'dRel6': [],
         'dRel8': [],
         'dRel10': [],
    }
    # PyTorch dRel history (after patch is applied)
    torch_optmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    # ONNX dRel history (before patch is applied)
    onnx_init_drel_hist = copy.deepcopy(torch_init_drel_hist)
    # ONNX dRel history (after patch is applied + optimized)
    onnx_optmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    # PyTorch dRel history (with random patch)
    torch_randmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    # ONNX dRel history (with random patch)
    onnx_randmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    # PyTorch dRel history (with FGSM attack)
    torch_fgsm_drel_hist = copy.deepcopy(torch_init_drel_hist)
    # ONNX dRel history (with FGSM attack)
    onnx_fgsm_drel_hist = copy.deepcopy(torch_init_drel_hist)



    rmse_hist = []
    frame_hist = []
    mask_effect_hist = []
    patch_h_hist = []
    patch_w_hist = []

    img_center = torch.tensor([[437, 582, 437, 582]]).float()
    thres = 3
    mask_iterations = 10
    # patches = []

    frame_idx = 0
    while(cap.isOpened()):
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(frame_idx)

        ret, frame = cap.read()
        if (ret == False):
            break

        # read in new frame
        if frame is not None:
            # read bounding box with YOLO
            results = yolo_model(frame, verbose=False)
            print('--- {} ---'.format(frame_idx))
            boxes = results[0].boxes

            # choose box closest to the center, most likely to be lead
            # min_dist = float('inf')
            # for i in range(len(boxes)):
            #     box_ = boxes[i].xyxy
            #     d = dist(box_, img_center)
            #     if d < min_dist:
            #         box = box_
            #         min_dist = d

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
            
            print(box)
            if isinstance(box, torch.Tensor):
                box = box.int().numpy()
            else:
                box = box.astype('int32')
            
            patch_start = (box[0, 1], box[0, 0])
            patch_dim = (box[0, 3] - box[0, 1], box[0, 2] - box[0, 0])
            patch_w_hist.append(patch_dim[0])
            patch_h_hist.append(patch_dim[1])
            print('Patch size:', patch_dim)
            
            # convert image to YUV and create patch
            img = cv2.resize(frame, dim)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).astype('float32')
            if len(proc_images) >= 2:
                del proc_images[0]
            proc_images.append(img_yuv)
            patch, h_bounds, w_bounds = build_yuv_patch(thres, patch_dim, patch_start)
            # img_yuv[h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += patch
            # cv2.imwrite('test.png', img_yuv.astype('uint8'))

            parsed = parse_image(img_yuv)

        # remove first frame and add new frame
        if (len(parsed_images) >= 2):
            del parsed_images[0]

        parsed_images.append(parsed)

        # supercombo model takes in past 2 frames: only run inference if 2 frames are in the buffer
        if (len(parsed_images) >= 2):
            
            # initial pass through model with no patch
            input_imgs = np.array(parsed_images).astype('float32')
            input_imgs.resize((1,12,128,256))
            
            torch_drel, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
            onnx_drel, onnx_rec_state = estimate_onnx(onnx_model, input_imgs, desire, traffic_convention, onnx_rec_state)
            
            # record dRel for ONNX and initial PyTorch
            torch_drel = np.array([v.detach().numpy() for v in torch_drel])
            for j, t in enumerate(pred_times):
                torch_init_drel_hist['dRel'+str(t)].append(torch_drel[j])
                onnx_init_drel_hist['dRel'+str(t)].append(onnx_drel[j])
            
            rmse_hist.append(np.sqrt(mean_squared_error(torch_drel, onnx_drel)))
            frame_hist.append(frame_idx)

            print('Lead car rel dist no patch: {}'.format(torch_drel[0]))

            # optimize patch
            patch = torch.tensor(patch, requires_grad=True)
            adam = AdamOptTorch(patch.shape, lr = 1)

            for it in range(mask_iterations):
                # conduct FGSM attack
                if it == 1:
                    tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
                    tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += fgsm_patch
                    tmp_pimgs = parse_image_multi(tmp_pimgs)
                    input_imgs = tmp_pimgs.reshape(1, 12, 128, 256)

                    torch_drel, _ = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
                    onnx_drel, _ = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
                    
                    print('Lead car rel dist with FGSM patch: {}'.format(torch_drel[0]))
                    torch_drel = np.array([v.detach().numpy() for v in torch_drel])
                    for j, t in enumerate(pred_times):
                        torch_fgsm_drel_hist['dRel'+str(t)].append(torch_drel[j])
                        onnx_fgsm_drel_hist['dRel'+str(t)].append(onnx_drel[j])

                # apply new patch to the image
                patch = torch.clip(patch, -thres, thres)
                tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
                tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += patch

                tmp_pimgs = parse_image_multi(tmp_pimgs)
                input_imgs = tmp_pimgs.reshape(1, 12, 128, 256)

                # pass newly masked image through the model
                torch_drel, _ = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
                
                # record stats for random mask
                if it == 0:
                    onnx_drel, _ = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
                    print('Lead car rel dist with random patch: {}'.format(onnx_drel[0]))
                    torch_drel_rand = np.array([v.detach().numpy() for v in torch_drel])

                    for j, t in enumerate(pred_times):
                        torch_randmask_drel_hist['dRel'+str(t)].append(torch_drel_rand[j])
                        onnx_randmask_drel_hist['dRel'+str(t)].append(onnx_drel[j])

                patch.retain_grad()
                torch_drel[0].backward()
                grad = patch.grad
                
                # prep fgsm patch after 1st pass through
                if it == 0:
                    fgsm_patch = patch + thres*torch.sign(grad)

                # update patch
                update = adam.update(grad)
                patch = patch.clone().detach().requires_grad_(True) + update
                torch_rec_state = torch_rec_state.clone().detach()

            # send input images with optimized patch through models
            onnx_drel, _ = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
            torch_drel, _ = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
            
            torch_drel = np.array([v.detach().numpy() for v in torch_drel])
            for j, t in enumerate(pred_times):
                onnx_optmask_drel_hist['dRel'+str(t)].append(onnx_drel[j])
                torch_optmask_drel_hist['dRel'+str(t)].append(torch_drel[j])

            print('Lead car rel dist after {} updates: {}'.format(mask_iterations, onnx_drel[0]))
            mask_effect_hist.append(onnx_drel[0] - torch_drel_rand[0])


        
    data = {}
    data['Frame'] = frame_hist
    for t in pred_times:
        data['Torch_init_dRel'+str(t)] = torch_init_drel_hist['dRel'+str(t)]
        data['Torch_randmask_dRel'+str(t)] = torch_randmask_drel_hist['dRel'+str(t)]
        data['Torch_optmask_dRel'+str(t)] = torch_optmask_drel_hist['dRel'+str(t)]
        data['Torch_fgsm_dRel'+str(t)] = torch_fgsm_drel_hist['dRel'+str(t)]
        data['ONNX_init_dRel'+str(t)] = onnx_init_drel_hist['dRel'+str(t)]
        data['ONNX_randmask_dRel'+str(t)] = onnx_randmask_drel_hist['dRel'+str(t)]
        data['ONNX_optmask_dRel'+str(t)] = onnx_optmask_drel_hist['dRel'+str(t)]
        data['ONNX_fgsm_dRel'+str(t)] = onnx_fgsm_drel_hist['dRel'+str(t)]
        
    data['ONNX_Torch_RMSE'] = rmse_hist
    data['Mask_dRel_Effect'] = mask_effect_hist
    data['Patch_Height'] = patch_h_hist
    data['Patch_Width'] = patch_w_hist

    
    df = pd.DataFrame(data=data)
    global chunk
    df.to_csv(f'./results/with_patch/chunk{chunk}_v4/video{video_id}.csv', index=False)

    cap.release()
    cv2.destroyAllWindows()

# def analyze_video_multi(video_paths, video_ids):
#     batch_size = len(video_paths)

#     onnx_model = onnxruntime.InferenceSession(r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_server3.onnx", providers=['CUDAExecutionProvider'])
#     torch_model = load_torch_model()
	
#     captures = []
#     for i in range(batch_size):
#         captures.append(cv2.VideoCapture(video_paths[i]))

#     parsed_images = [[] for _ in range(batch_size)]

#     width = 512
#     height = 256
#     dim = (width, height)

#     desire = np.zeros(shape=(batch_size, 8)).astype('float32')
#     traffic_convention = np.repeat([[1, 0]], axis=0, repeats=batch_size).astype('float32')
#     torch_rec_state = np.zeros(shape=(batch_size, 512)).astype('float32')
#     onnx_rec_state = np.zeros(shape=(batch_size, 512)).astype('float32')

#     pred_times = [0, 2, 4, 6, 8, 10]
#     torch_drel_hist = [{
#          'dRel0': [],
#          'dRel2': [],
#          'dRel4': [],
#          'dRel6': [],
#          'dRel8': [],
#          'dRel10': [],
#     } for _ in range(batch_size)]
#     onnx_drel_hist = [{
#          'dRel0': [],
#          'dRel2': [],
#          'dRel4': [],
#          'dRel6': [],
#          'dRel8': [],
#          'dRel10': [],
#     } for _ in range(batch_size)]
#     rmse_hist = [[] for _ in range(batch_size)]
#     frame_hist = [[] for _ in range(batch_size)]

#     i = 0
#     # while(cap.isOpened()):
#     while i < 1200:
#         i += 1
#         if i % 200 == 0:
#             print(i)

#         for i in range(batch_size):
#             cap = captures[i]
#             ret, frame = cap.read()
#             if (ret == False):
#                 break

#             # read in new frame
#             if frame is not None:
#                 img = cv2.resize(frame, dim)
#                 img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
#                 parsed = parse_image(img_yuv)

#             # remove first frame and add new frame
#             if (len(parsed_images[i]) >= 2):
#                 del parsed_images[i][0]

#             parsed_images[i].append(parsed)

#         # supercombo model takes in past 2 frames: only run inference if 2 frames are in the buffer
#         if (len(parsed_images[0]) >= 2):
#             # parsed_images: N x 2 x 6 x 128 x 256
#             input_imgs = np.array(parsed_images).astype('float32')
#             input_imgs.resize((batch_size , 12, 128, 256))
            
#             torch_lp, torch_drel, torch_speed, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
#             # onnx_lp, onnx_drel, onnx_speed, onnx_rec_state = estimate_onnx(onnx_model, input_imgs, desire, traffic_convention, onnx_rec_state)

#             # if torch_drel.shape[0] > 1:
#             #     print('Torch dRel has wrong shape: ', torch_drel.shape[0])
#             #     break
#             # if onnx_drel.shape[0] > 1:
#             #     print('ONNX dRel has wrong shape: ', onnx_drel.shape[0])
#             #     break

#             for k in range(batch_size):
#                 for j, t in enumerate(pred_times):
#                     torch_drel_hist[k]['dRel'+str(t)].append(torch_drel[k, j])
#                     # onnx_drel_hist[k]['dRel'+str(t)].append(onnx_drel[k, j])
                
#                 # rmse_hist[k].append(np.sqrt(mean_squared_error(torch_drel[k, :], onnx_drel[k, :])))
#                 frame_hist[k].append(i)
    
#     for k in range(batch_size):
#         data = {}
#         data['Frame'] = frame_hist[k]
#         for t in pred_times:
#             data['Torch_dRel'+str(t)] = torch_drel_hist[k]['dRel'+str(t)]
#             # data['ONNX_dRel'+str(t)] = onnx_drel_hist[k]['dRel'+str(t)]
#         # data['RMSE'] = rmse_hist[k]

        
#         df = pd.DataFrame(data=data)
#         df.to_csv(f'../results/chunk2_multi_video{video_ids[k]}.csv', index=False)

#         cap = captures[k]
#         cap.release()
#         cv2.destroyAllWindows()


def collect_results_rmse(res_path, name_filter='', patch_type='init'):
    all_mse = []

    indiv_mse = {}
    pred_times = [0, 2, 4, 6, 8, 10]
    for t in pred_times:
        indiv_mse['dRel'+str(t)] = []

    dist_mse = {}
    bounds = [0, 20, 40, 60, 80, 999]
    torch_cols = [f'Torch_{patch_type}_dRel{t}' for t in pred_times]
    onnx_cols = [f'ONNX_{patch_type}_dRel{t}' for t in pred_times]
    for i in range(1, len(bounds)):
        dist_mse[bounds[i]] = []

    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith('.csv') and name_filter in file:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                torch_preds = df.loc[:, torch_cols].to_numpy()
                onnx_preds = df.loc[:, onnx_cols].to_numpy()
                trace_mse = np.mean(np.power(torch_preds - onnx_preds, 2))
                all_mse.append(trace_mse)
                # gather MSE by prediction horizon
                for t in pred_times:
                    pred_diff = df[f'Torch_{patch_type}_dRel{t}'] - df[f'ONNX_{patch_type}_dRel{t}']
                    pred_mse = pred_diff.pow(2).mean()
                    indiv_mse['dRel'+str(t)].append(pred_mse)
                # gather MSE by distance estimate
                for i in range(1, len(bounds)):
                    pred_within_bounds = np.all([df[torch_cols] >= bounds[i-1], df[torch_cols] <= bounds[i]], axis=0)
                    # check = np.all([df[onnx_cols] >= bounds[i-1], df[onnx_cols] <= bounds[i]], axis=0)
                    # # if not np.all(pred_within_bounds == check):
                    # #     print('Edge case for UB', bounds[i], 'in file', file_path)
                    # #     print('\t', np.where((pred_within_bounds == check) == False))
                    torch_mask = np.array(df[torch_cols]*pred_within_bounds)
                    onnx_mask = np.array(df[onnx_cols]*pred_within_bounds)
                    pred_mse = np.mean(np.power(torch_mask - onnx_mask, 2))
                    dist_mse[bounds[i]].append(pred_mse)
    
    rmse = np.sqrt(np.mean(all_mse))
    indiv_rmse = {}
    for t in pred_times:
        indiv_rmse['dRel'+str(t)] = np.sqrt(np.mean(indiv_mse['dRel'+str(t)]))

    dist_rmse = {}
    for i in range(1, len(bounds)):
        dist_rmse[bounds[i]] = np.sqrt(np.mean(dist_mse[bounds[i]]))

    return rmse, indiv_rmse, dist_rmse

def collect_results_patched(res_path, blacklist, name_filter=''):
    pred_times = [0, 2, 4, 6, 8, 10]
    
    opt_avg_dev, rand_avg_dev, fgsm_avg_dev = [], [], []
    opt_max_dev, rand_max_dev, fgsm_max_dev = 0, 0, 0
    opt_avg_dev_by_dist, rand_avg_dev_by_dist, fgsm_avg_dev_by_dist = {}, {}, {}
    opt_max_dev_by_dist, rand_max_dev_by_dist, fgsm_max_dev_by_dist = {}, {}, {}
    bounds = [0, 20, 40, 60, 80, 999]
    # bounds = [0, 5, 10, 15, 20, 40, 60, 80, 999]
    # torch_init_cols = ['Torch_init_dRel'+str(t) for t in pred_times]
    # onnx_init_cols = ['ONNX_init_dRel'+str(t) for t in pred_times]
    # onnx_randmask_cols = ['ONNX_randmask_dRel'+str(t) for t in pred_times]
    # onnx_optmask_cols = ['ONNX_optmask_dRel'+str(t) for t in pred_times]
    # onnx_fgsm_cols = ['ONNX_fgsm_dRel'+str(t) for t in pred_times]


    for i in range(1, len(bounds)):
        opt_avg_dev_by_dist[bounds[i]] = []
        rand_avg_dev_by_dist[bounds[i]] = []
        fgsm_avg_dev_by_dist[bounds[i]] = []
        
        opt_max_dev_by_dist[bounds[i]] = 0
        rand_max_dev_by_dist[bounds[i]] = 0
        fgsm_max_dev_by_dist[bounds[i]] = 0

    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith('.csv') and name_filter in file and file not in blacklist:
                file_path = os.path.join(root, file)

                df = pd.read_csv(file_path)
                df = df.iloc[10:] # remove first 10 frames as recurrent state isn't stable

                opt_effect = df['ONNX_optmask_dRel0'] - df['ONNX_init_dRel0']
                rand_effect = df['ONNX_randmask_dRel0'] - df['ONNX_init_dRel0']
                fgsm_effect = df['ONNX_fgsm_dRel0'] - df['ONNX_init_dRel0']

                # record overall max and avg dev
                opt_avg_dev.append(opt_effect.mean())
                rand_avg_dev.append(rand_effect.mean())
                fgsm_avg_dev.append(fgsm_effect.mean())
                
                opt_max_dev = max(opt_max_dev, opt_effect.max())
                rand_max_dev = max(rand_max_dev, rand_effect.max())
                fgsm_max_dev = max(fgsm_max_dev, fgsm_effect.max())

                # return avg and max deviation in dRel0 caused by patch
                for i in range(1, len(bounds)):
                    pred_within_bounds = np.all([df['ONNX_init_dRel0'] >= bounds[i-1], df['ONNX_init_dRel0'] <= bounds[i]], axis=0)
                    opt_dev_within_bounds = opt_effect[pred_within_bounds].to_numpy()
                    rand_dev_within_bounds = rand_effect[pred_within_bounds].to_numpy()
                    fgsm_dev_within_bounds = fgsm_effect[pred_within_bounds].to_numpy()
                    
                    if len(opt_dev_within_bounds) == 0:
                        continue
                    
                    opt_avg_dev_within_bounds = np.mean(opt_dev_within_bounds)
                    rand_avg_dev_within_bounds = np.mean(rand_dev_within_bounds)
                    fgsm_avg_dev_within_bounds = np.mean(fgsm_dev_within_bounds)
                    
                    opt_max_dev_within_bounds = np.max(opt_dev_within_bounds)
                    rand_max_dev_within_bounds = np.max(rand_dev_within_bounds)
                    fgsm_max_dev_within_bounds = np.max(fgsm_dev_within_bounds)

                    opt_avg_dev_by_dist[bounds[i]].append(opt_avg_dev_within_bounds)
                    rand_avg_dev_by_dist[bounds[i]].append(rand_avg_dev_within_bounds)
                    fgsm_avg_dev_by_dist[bounds[i]].append(fgsm_avg_dev_within_bounds)
                    
                    if opt_max_dev_within_bounds > opt_max_dev_by_dist[bounds[i]]:
                        opt_max_dev_by_dist[bounds[i]] = opt_max_dev_within_bounds
                    if rand_max_dev_within_bounds > rand_max_dev_by_dist[bounds[i]]:
                        rand_max_dev_by_dist[bounds[i]] = rand_max_dev_within_bounds
                    if fgsm_max_dev_within_bounds > fgsm_max_dev_by_dist[bounds[i]]:
                        fgsm_max_dev_by_dist[bounds[i]] = fgsm_max_dev_within_bounds
                    if rand_max_dev_within_bounds > 20:
                        print(rand_max_dev_within_bounds, bounds[i-1], bounds[i], file_path, len(rand_dev_within_bounds))
                        print(rand_dev_within_bounds)
                        idx = np.where(pred_within_bounds == True)
                        print(idx)
                        print(df.loc[pred_within_bounds, ['ONNX_init_dRel0', 'ONNX_randmask_dRel0', 'Patch_Height', 'Patch_Width']])
    
    opt_avg_dev_by_dist_collected, rand_avg_dev_by_dist_collected, fgsm_avg_dev_by_dist_collected = {}, {}, {}
    for i in range(1, len(bounds)):
        opt_avg_dev_by_dist_collected[bounds[i]] = np.mean(opt_avg_dev_by_dist[bounds[i]])
        rand_avg_dev_by_dist_collected[bounds[i]] = np.mean(rand_avg_dev_by_dist[bounds[i]])
        fgsm_avg_dev_by_dist_collected[bounds[i]] = np.mean(fgsm_avg_dev_by_dist[bounds[i]])

    opt_res = (np.mean(opt_avg_dev), opt_max_dev, opt_avg_dev_by_dist_collected, opt_max_dev_by_dist)
    rand_res = (np.mean(rand_avg_dev), rand_max_dev, rand_avg_dev_by_dist_collected, rand_max_dev_by_dist)
    fgsm_res = (np.mean(fgsm_avg_dev), fgsm_max_dev, fgsm_avg_dev_by_dist_collected, fgsm_max_dev_by_dist)

    return opt_res, rand_res, fgsm_res

# measures deviation from unmasked predictions
def get_deviation_stats(res_path, blacklist, col_type='optmask'):
    bounds = [0, 20, 40, 60, 80, 999]

    init_predictions = None
    mask_predictions = None

    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith('.csv') and file not in blacklist:
                file_path = os.path.join(root, file)

                df = pd.read_csv(file_path)
                df = df.iloc[10:] # remove first 10 frames as recurrent state isn't stable

                if init_predictions is None:
                    init_predictions = df['ONNX_init_dRel0'].to_numpy()
                    mask_predictions = df[f'ONNX_{col_type}_dRel0'].to_numpy()
                else:
                    init_predictions = np.r_[init_predictions, df['ONNX_init_dRel0'].to_numpy()]
                    mask_predictions = np.r_[mask_predictions, df[f'ONNX_{col_type}_dRel0'].to_numpy()]
    
    deviation = mask_predictions - init_predictions
    avg_dev = np.mean(deviation)
    std_dev = np.std(deviation)

    avg_dev_by_dist = {}
    std_dev_by_dist = {}

    for i in range(1, len(bounds)):
        pred_within_bounds = np.all([init_predictions >= bounds[i-1], init_predictions <= bounds[i]], axis=0)
        deviation_within_bounds = deviation[pred_within_bounds]

        avg_dev_by_dist[bounds[i]] = np.mean(deviation_within_bounds)
        std_dev_by_dist[bounds[i]] = np.std(deviation_within_bounds)

    return avg_dev, std_dev, avg_dev_by_dist, std_dev_by_dist

if __name__ == '__main__':
    video_blacklist = [
        '/home/student/Max/Chunk_3/99c94dc769b5d96e|2018-05-18--10-58-35/24/video.hevc',
    ]
    result_blacklist = [
        'video6.csv',
    ]

    chunk = 3
    video_id = 0

    # if not os.path.exists(f'./results/with_patch/chunk{chunk}_v3/filepaths.txt'):
    #     with open(f'./results/with_patch/chunk{chunk}_v4/filepaths.txt', 'x') as f:
    #         f.write('video_id|path\n')

    for root, dirs, files in os.walk(f'/home/student/Max/Chunk_{chunk}/'):
        for file in files:
            if file.endswith('.hevc'):
                video_path = os.path.join(root, file)
        #            idx = root.find('\\')
        #            video_id = root[idx-20:idx] + '--' + root[idx+1:]
                video_id += 1
                if video_id <= 8 or video_path in video_blacklist:# and video_id <= 8:
                    continue

                # record path of video
                with open(f'./results/with_patch/chunk{chunk}_v4/filepaths.txt', 'a') as f:
                    f.write(str(video_id) + '|' + str(video_path) + '\n')

                print('Starting video', video_id)
                start = datetime.now()
                analyze_video(video_path, video_id)
                dur = datetime.now() - start
                print(dur, 'for file', video_id)
                

    #             if video_id >= 8:
    #                 print('END OF SUBSET')
    #                 break
    #     if video_id >= 8:
    #         break

    ### collect results ###
    # res = collect_results_patched(f'/home/student/Max/results/with_patch/chunk{chunk}_v4', result_blacklist, name_filter='')

    for i in range(3):
        if i == 0:
            print('### Optimized Patch ###')
            pt = 'optmask'
        if i == 1:
            print('### Random Patch ###')
            pt = 'randmask'
        if i == 2:
            print('### FGSM Patch ###')
            pt = 'fgsm'
        
        avg_dev, std_dev, avg_dev_by_dist, std_dev_by_dist = get_deviation_stats(f'/home/student/Max/results/with_patch/chunk{chunk}_v3', result_blacklist, col_type=pt)

        print(f'Overall: {avg_dev} (avg), {std_dev} (std)')

        bounds = [0, 20, 40, 60, 80, 999]
        # bounds = [0, 5, 10, 15, 20, 40, 60, 80, 999]
        for i in range(1, len(bounds)):
            print(f"[{bounds[i-1]}, {bounds[i]}] : {avg_dev_by_dist[bounds[i]]} (avg), {std_dev_by_dist[bounds[i]]}, (std)")

        print()
    #     rmse, indiv_rmse, dist_rmse = collect_results_rmse(f'/home/student/Max/results/with_patch/chunk{chunk}_v4', name_filter='', patch_type=pt)
    #     print(rmse)
    #     print(dist_rmse)

    # rmse, indiv_rmse, dist_rmse = collect_results_rmse(f'/home/student/Max/results/with_patch/chunk{chunk}_v4', name_filter='', patch_type='init')
    # print()
    # print(rmse)
    # print(dist_rmse)