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

PYTORCH_MODEL_PATH = r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_torch_weights.pth"
ONNX_MODEL_PATH = r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_server3.onnx"

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
    rec_state = out[:, 6097:]

    return lead_prob, d_rel, speed, rec_state

def estimate_onnx(model, input_imgs, desire, traffic_convention, recurrent_state):
    out = model.run(['outputs'], {
        'input_imgs': input_imgs,
        'desire': desire,
        'traffic_convention': traffic_convention,
        'initial_state': recurrent_state
    })

    out = np.array(out[0])

    lead_prob = sigmoid(out[:, 6010:6013])
    x_rel = out[:, 5755:5779:4]
    y_rel = out[:, 5755+1:5779:4]
    d_rel = np.sqrt(np.power(x_rel, 2) + np.power(y_rel, 2))

    speed = out[:, 5755+2:5779:4]

    rec_state = out[:, 6097:]

    return lead_prob, d_rel, speed, rec_state


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

def load_torch_model():
    model = OpenPilotModel()
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
    model.eval()

    return model

def analyze_video(video_path, video_id):

    onnx_model = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
    torch_model = load_torch_model()
	
    cap = cv2.VideoCapture(video_path)
    parsed_images = []

    width = 512
    height = 256
    dim = (width, height)

    desire = np.zeros(shape=(1, 8)).astype('float32')
    traffic_convention = np.array([[1, 0]]).astype('float32')
    torch_rec_state = np.zeros(shape=(1, 512)).astype('float32')
    onnx_rec_state = np.zeros(shape=(1, 512)).astype('float32')

    pred_times = [0, 2, 4, 6, 8, 10]
    torch_drel_hist = {
         'dRel0': [],
         'dRel2': [],
         'dRel4': [],
         'dRel6': [],
         'dRel8': [],
         'dRel10': [],
    }
    onnx_drel_hist = {
         'dRel0': [],
         'dRel2': [],
         'dRel4': [],
         'dRel6': [],
         'dRel8': [],
         'dRel10': [],
    }
    rmse_hist = []
    frame_hist = []

    i = 0
    while(cap.isOpened()):
        i += 1
        if i % 200 == 0:
            print(i)

        ret, frame = cap.read()
        if (ret == False):
            break

        # read in new frame
        if frame is not None:
            img = cv2.resize(frame, dim)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            parsed = parse_image(img_yuv)

        # remove first frame and add new frame
        if (len(parsed_images) >= 2):
            del parsed_images[0]

        parsed_images.append(parsed)

        # supercombo model takes in past 2 frames: only run inference if 2 frames are in the buffer
        if (len(parsed_images) >= 2):
        
            input_imgs = np.array(parsed_images).astype('float32')
            input_imgs.resize((1,12,128,256))
            
            torch_lp, torch_drel, torch_speed, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
            onnx_lp, onnx_drel, onnx_speed, onnx_rec_state = estimate_onnx(onnx_model, input_imgs, desire, traffic_convention, onnx_rec_state)

            if torch_drel.shape[0] > 1:
                print('Torch dRel has wrong shape: ', torch_drel.shape[0])
                break
            if onnx_drel.shape[0] > 1:
                print('ONNX dRel has wrong shape: ', onnx_drel.shape[0])
                break

            for j, t in enumerate(pred_times):
                torch_drel_hist['dRel'+str(t)].append(torch_drel[0, j])
                onnx_drel_hist['dRel'+str(t)].append(onnx_drel[0, j])
            
            rmse_hist.append(np.sqrt(mean_squared_error(torch_drel, onnx_drel)))
            frame_hist.append(i)
        
    data = {}
    data['Frame'] = frame_hist
    for t in pred_times:
        data['Torch_dRel'+str(t)] = torch_drel_hist['dRel'+str(t)]
        data['ONNX_dRel'+str(t)] = onnx_drel_hist['dRel'+str(t)]
    data['RMSE'] = rmse_hist

    
    df = pd.DataFrame(data=data)
    df.to_csv(f'../results/chunk2_video{video_id}.csv', index=False)

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


def collect_results(res_path):
    all_rmse = []

    indiv_rmse = {}
    pred_times = [0, 2, 4, 6, 8, 10]
    for t in pred_times:
        indiv_rmse['dRel'+str(t)] = []

    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                trace_rmse = (1/len(df))*df['RMSE'].sum()
                all_rmse.append(trace_rmse)
                for t in pred_times:
                    indiv_rmse['dRel'+str(t)].append((1/len(df))*df['dRel'+str(t)].sum())

    
    rmse = (1/len(all_rmse))*np.sum(all_rmse)
    for t in pred_times:
        indiv_rmse['dRel'+str(t)] = (1/len(all_rmse))*np.sum(indiv_rmse['dRel'+str(t)])

    return rmse, indiv_rmse


def save_test_inputs(video_path):
    onnx_model = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
    torch_model = load_torch_model()
	
    cap = cv2.VideoCapture(video_path)
    parsed_images = []
    imgs = []

    width = 512
    height = 256
    dim = (width, height)

    desire = np.zeros(shape=(1, 8)).astype('float32')
    traffic_convention = np.array([[1, 0]]).astype('float32')
    torch_rec_state = np.zeros(shape=(1, 512)).astype('float32')
    onnx_rec_state = np.zeros(shape=(1, 512)).astype('float32')

    i = 0
    while(cap.isOpened()):
        i += 1
        if i % 200 == 0:
            print(i)

        ret, frame = cap.read()
        if (ret == False):
            break

        # read in new frame
        if frame is not None:
            img = cv2.resize(frame, dim)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            parsed = parse_image(img_yuv)

        # remove first frame and add new frame
        if (len(parsed_images) >= 2):
            del parsed_images[0]
            del imgs[0]

        parsed_images.append(parsed)
        imgs.append(frame)

        # supercombo model takes in past 2 frames: only run inference if 2 frames are in the buffer
        if (len(parsed_images) >= 2):
        
            input_imgs = np.array(parsed_images).astype('float32')
            input_imgs.resize((1,12,128,256))
            
            # save inputs
            np.save('test_inputs/input_imgs', input_imgs)
            np.save('test_inputs/desire', desire)
            np.save('test_inputs/traffic_convention', traffic_convention)
            np.save('test_inputs/initial_state', torch_rec_state)
            np.save('test_inputs/original_image1', imgs[0])
            np.save('test_inputs/original_image2', imgs[1])
        
            break
        
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    # video_id = 0

    # for root, dirs, files in os.walk('E:/Chunk_2/'):
    #     for file in files:
    #         if file.endswith('.hevc'):
    #             video_path = os.path.join(root, file)
    #             idx = root.find('\\')
    #             video_id = root[idx-20:idx] + '--' + root[idx+1:]

    #             start = datetime.now()
    #             analyze_video(video_path, video_id)
    #             dur = datetime.now() - start
    #             print(dur, 'for file', video_id)
                

    #             # if video_id >= 5:
    #             sys.exit(0)

    # rmse, indiv_rmse = collect_results('/home/student/Max/results/')
    # print('RMSE across all files', rmse)

    # for t in [0, 2, 4, 6, 8, 10]:
    #     print(f'RMSE for prediction horizon of {t} s:', indiv_rmse['dRel'+str(t)])

    save_test_inputs('../dataset/video.hevc')