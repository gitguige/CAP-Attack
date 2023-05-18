import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from openpilot_torch import OpenPilotModel
import random
import time

PYTORCH_MODEL_PATH = r"C:\Users\Max\Documents\UVA Docs\Second Year Classes\Research\ADS\openpilot_model\supercombo_torch_weights.pth"

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
    

def process_image(img):
	if img.dtype == np.float32:
		img = np.clip(img, 0, 255).astype('uint8')
	img = cv2.resize(img, (512, 256))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
	return img

def parse_image(frame):
    H = (frame.shape[0]*2)//3
    W = frame.shape[1]

    if isinstance(frame, torch.Tensor):
        parsed = torch.zeros(size=(6, H//2, W//2)).float()
    else:
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
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
    model.eval()

    return model

def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return 1 / (1 + torch.exp(-x))
    return 1 / (1 + np.exp(-x))

def estimate_torch_grad(model, input_imgs, desire, traffic_convention, recurrent_state):
    if not isinstance(desire, torch.Tensor):
        # input_imgs = torch.tensor(input_imgs).float()
        desire = torch.tensor(desire).float()
        traffic_convention = torch.tensor(traffic_convention).float()
        recurrent_state = torch.tensor(recurrent_state).float()
    
    out = model(input_imgs, desire, traffic_convention, recurrent_state)
    print(out[0, 5750:5760])

    lead_prob = sigmoid(out[:, 6010:6013])
    x_rel = out[:, 5755:5779:4]
    y_rel = out[:, 5755+1:5779:4]
    d_rel = torch.sqrt(torch.pow(x_rel, 2) + torch.pow(y_rel, 2))

    speed = out[:, 5755+2:5779:4]
    rec_state = out[:, 6097:]

    return lead_prob, d_rel, speed, rec_state

def estimate_torch(model, input_imgs, desire, traffic_convention, recurrent_state):
    if not isinstance(input_imgs, torch.Tensor):
        input_imgs = torch.tensor(input_imgs).float()
        desire = torch.tensor(desire).float()
        traffic_convention = torch.tensor(traffic_convention).float()
        recurrent_state = torch.tensor(recurrent_state).float()
    
    out = model(input_imgs, desire, traffic_convention, recurrent_state).detach().numpy()
    print(out[0, 5750:5760])

    lead_prob = sigmoid(out[:, 6010:6013])
    x_rel = out[:, 5755:5779:4]
    y_rel = out[:, 5755+1:5779:4]
    d_rel = np.sqrt(np.power(x_rel, 2) + np.power(y_rel, 2))

    speed = out[:, 5755+2:5779:4]
    rec_state = out[:, 6097:]

    return lead_prob, d_rel, speed, rec_state

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

def build_yuv_patch(thres):
    h_ratio = 256/874
    w_ratio = 512/1174
    y_patch_height = int(patch_height*h_ratio)
    y_patch_width = int(patch_width*w_ratio)
    y_patch_h_start = int(350*h_ratio)
    y_patch_w_start = int(100*w_ratio)

    y_patch = thres * np.random.rand(y_patch_height, y_patch_width).astype('float32')
    # u_patch = thres * np.random.rand()
    h_bounds = (y_patch_h_start, y_patch_h_start+y_patch_height)
    w_bounds = (y_patch_w_start, y_patch_w_start+y_patch_width)
    return y_patch, h_bounds, w_bounds


if __name__ == '__main__':
    img_path = r"E:\golden-left-75m\golden-left-75m\imgs\660.png"
    # img2_path = 
    sample_image = cv2.imread(img_path)
    imgs = [sample_image.astype('float32'), sample_image.astype('float32')]

    patch_height = 200
    patch_width = 900
    thres = 1
    iterations = 10

    # grad = np.zeros((patch_height, patch_width, 3)).astype('float32')
    # prev_patch = thres * np.random.rand(patch_height, patch_width, 3).astype('float32')
    # patch = thres * np.random.rand(patch_height, patch_width, 3).astype('float32')
    # patch = thres * torch.rand(size=(patch_height, patch_width, 3), requires_grad=True).float()
    proc_images = [process_image(img) for img in imgs]
    parsed_images = [parse_image(img) for img in proc_images]
    cv2.imwrite('images/original.png', proc_images[0])

    desire = np.zeros(shape=(1, 8)).astype('float32')
    desire[0, 0] = 1
    traffic_convention = np.array([[1, 0]]).astype('float32')
    rec_state = np.zeros(shape=(1, 512)).astype('float32')
    model = load_torch_model()


    # initialize prev_prob
    tmp_imgs = copy.deepcopy(imgs)
    # tmp_imgs[0][350:350+patch_height,100:100+patch_width] += prev_patch
    # tmp_imgs[1][350:350+patch_height,100:100+patch_width] += prev_patch
    # proc_images = []
    # parsed_images[0] = parse_image(process_image(tmp_imgs[0]))
    # parsed_images[1] = parse_image(process_image(tmp_imgs[1]))
    input_imgs = np.array(parsed_images).reshape(1, 12, 128, 256)
    lead_prob, d_rel, speed, _ = estimate_torch(model, input_imgs, desire, traffic_convention, rec_state)
    prev_drel0 = d_rel[0, 0]
    print('Rel dist of lead car with no patch:', prev_drel0)
    patch, h_bounds, w_bounds = build_yuv_patch(thres)
    patch = torch.tensor(patch, requires_grad=True)

    adam = AdamOptTorch(patch.shape, lr = 1)

    tic = time.time()
    for it in range(iterations):
        # apply new patch to the image
        patch = torch.clip(patch, -thres, thres)
        tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
        tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += patch

        parsed_images = parse_image_multi(tmp_pimgs)
        input_imgs = parsed_images.reshape(1, 12, 128, 256)

        # pass newly masked image through the model
        lead_prob, d_rel, speed, _ = estimate_torch_grad(model, input_imgs, desire, traffic_convention, rec_state)
        drel0 = d_rel[0, 0]
        
        print('Lead car rel dist after {} updates: {}'.format(it, drel0))
        patch.retain_grad()
        drel0.backward()
        grad = patch.grad
        
        # update patch
        update = adam.update(grad)
        patch = patch.clone().detach().requires_grad_(True) + update

        masked_img = tmp_pimgs[0].detach().numpy()
        cv2.imwrite('images/masked_image{}.png'.format(it), masked_img)
        
    dur = time.time() - tic
    avg_time = dur/iterations
    print(f'Took {dur:.4f} s for {iterations} iterations ({avg_time:.4f} s / iter)')