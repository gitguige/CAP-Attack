import torch
from torch import nn
from collections import namedtuple
import onnx
from onnx import numpy_helper
import onnxruntime
import numpy as np

ConvSettings = namedtuple("ConvSettings", "in_channels out_channels groups kernel_size stride padding dilation")

class VisionBlock(nn.Module):
    def __init__(self, depth_conv, proj_conv, expand_conv=None):
        super().__init__()

        self._depthwise_conv = nn.Conv2d(
            in_channels=depth_conv.in_channels, 
            out_channels=depth_conv.out_channels, 
            groups=depth_conv.groups, 
            kernel_size=depth_conv.kernel_size, 
            stride=depth_conv.stride, 
            padding=depth_conv.padding, 
            dilation=depth_conv.dilation,
            bias=False
        )
        self._project_conv = nn.Conv2d(
            in_channels=proj_conv.in_channels, 
            out_channels=proj_conv.out_channels, 
            groups=proj_conv.groups, 
            kernel_size=proj_conv.kernel_size, 
            stride=proj_conv.stride, 
            padding=proj_conv.padding, 
            dilation=proj_conv.dilation,
            bias=False
        )
        self._bn1 = nn.BatchNorm2d(depth_conv.out_channels, eps=0.001, momentum=0.99)
        self._bn2 = nn.BatchNorm2d(proj_conv.out_channels, eps=0.001, momentum=0.99)
        
        if expand_conv is not None:
            self._expand_conv = nn.Conv2d(
                in_channels=expand_conv.in_channels, 
                out_channels=expand_conv.out_channels, 
                groups=expand_conv.groups, 
                kernel_size=expand_conv.kernel_size, 
                stride=expand_conv.stride, 
                padding=expand_conv.padding, 
                dilation=expand_conv.dilation,
                bias=False
            )
            self._bn0 = nn.BatchNorm2d(expand_conv.out_channels, eps=0.001, momentum=0.99)
        else:
            self._expand_conv = None
            self._bn0 = None
        
        self.elu = nn.ELU(alpha=1.0, inplace=False)
        

    def forward(self, x):
        if self._expand_conv is not None:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.elu(x)
        
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.elu(x)
        x = self._project_conv(x)
        x = self._bn2(x)
        
        return x
    
class VisionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self._conv_stem = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=(3, 3),
            groups=1,
            padding=1,
            stride=2,
            bias=False
        )
        self._bn0 = nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.elu = nn.ELU(alpha=1.0, inplace=False)

        self.block0 = VisionBlock(
            depth_conv=ConvSettings(32, 32, groups=32, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(32, 16, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block1 = VisionBlock(
            depth_conv=ConvSettings(16, 16, groups=16, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(16, 16, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block2 = VisionBlock(
            expand_conv=ConvSettings(16, 96, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(96, 96, groups=96, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            proj_conv=ConvSettings(96, 24, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block3 = VisionBlock(
            expand_conv=ConvSettings(24, 144, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(144, 144, groups=144, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(144, 24, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block4 = VisionBlock(
            expand_conv=ConvSettings(24, 144, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(144, 144, groups=144, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(144, 24, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block5 = VisionBlock(
            expand_conv=ConvSettings(24, 144, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(144, 144, groups=144, kernel_size=(5, 5), stride=2, padding=2, dilation=1),
            proj_conv=ConvSettings(144, 48, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block6 = VisionBlock(
            expand_conv=ConvSettings(48, 288, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(288, 288, groups=288, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(288, 48, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block7 = VisionBlock(
            expand_conv=ConvSettings(48, 288, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(288, 288, groups=288, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(288, 48, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block8 = VisionBlock(
            expand_conv=ConvSettings(48, 288, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(288, 288, groups=288, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            proj_conv=ConvSettings(288, 88, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block9 = VisionBlock(
            expand_conv=ConvSettings(88, 528, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(528, 528, groups=528, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(528, 88, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block10 = VisionBlock(
            expand_conv=ConvSettings(88, 528, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(528, 528, groups=528, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(528, 88, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block11 = VisionBlock(
            expand_conv=ConvSettings(88, 528, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(528, 528, groups=528, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(528, 88, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block12 = VisionBlock(
            expand_conv=ConvSettings(88, 528, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(528, 528, groups=528, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(528, 120, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block13 = VisionBlock(
            expand_conv=ConvSettings(120, 720, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(720, 720, groups=720, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(720, 120, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block14 = VisionBlock(
            expand_conv=ConvSettings(120, 720, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(720, 720, groups=720, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(720, 120, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block15 = VisionBlock(
            expand_conv=ConvSettings(120, 720, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(720, 720, groups=720, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(720, 120, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block16 = VisionBlock(
            expand_conv=ConvSettings(120, 720, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(720, 720, groups=720, kernel_size=(5, 5), stride=2, padding=2, dilation=1),
            proj_conv=ConvSettings(720, 208, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block17 = VisionBlock(
            expand_conv=ConvSettings(208, 1248, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(1248, 1248, groups=1248, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(1248, 208, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block18 = VisionBlock(
            expand_conv=ConvSettings(208, 1248, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(1248, 1248, groups=1248, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(1248, 208, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block19 = VisionBlock(
            expand_conv=ConvSettings(208, 1248, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(1248, 1248, groups=1248, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(1248, 208, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block20 = VisionBlock(
            expand_conv=ConvSettings(208, 1248, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(1248, 1248, groups=1248, kernel_size=(5, 5), stride=1, padding=2, dilation=1),
            proj_conv=ConvSettings(1248, 208, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block21 = VisionBlock(
            expand_conv=ConvSettings(208, 1248, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(1248, 1248, groups=1248, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(1248, 352, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )
        self.block22 = VisionBlock(
            expand_conv=ConvSettings(352, 2112, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            depth_conv=ConvSettings(2112, 2112, groups=2112, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            proj_conv=ConvSettings(2112, 352, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )

        self._conv_head = VisionBlock(
            depth_conv=ConvSettings(352, 352, groups=352, kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            proj_conv=ConvSettings(352, 32, groups=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1)
        )



    def forward(self, x):
        # x = input_imgs: (1, 12, 128, 256)
        # pass x through stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self.elu(x)

        # main network body
        x = self.block0(x)
        y = self.block1(x)
        x = x + y

        x = self.block2(x)
        y = self.block3(x)
        x = x + y
        y = self.block4(x)
        x = x + y

        x = self.block5(x)
        y = self.block6(x)
        x = x + y
        y = self.block7(x)
        x = x + y

        x = self.block8(x)
        y = self.block9(x)
        x = x + y
        y = self.block10(x)
        x = x + y
        y = self.block11(x)
        x = x + y

        x = self.block12(x)
        y = self.block13(x)
        x = x + y
        y = self.block14(x)
        x = x + y
        y = self.block15(x)
        x = x + y

        x = self.block16(x)
        y = self.block17(x)
        x = x + y
        y = self.block18(x)
        x = x + y
        y = self.block19(x)
        x = x + y
        y = self.block20(x)
        x = x + y

        x = self.block21(x)
        y = self.block22(x)
        x = x + y

        # send x through head and out
        x = self._conv_head(x)
        return x.flatten(start_dim=1)

class TemporalSummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU(alpha=1.0, inplace=False)
        self.relu = nn.ReLU(inplace=False)

        self._infeats0 = nn.Linear(in_features=1034, out_features=1024, bias=True)
        self._outfeats0 = nn.Linear(in_features=1536, out_features=1024, bias=True)
        self._rnn = nn.GRU(input_size=1024, hidden_size=512, batch_first=True)

    def forward(self, vision, desire, traffic_convention, initial_state):
        # vision: 1 x 256, desire: 1 x 8, traffic_convention: 1 x 2, initial_state: 1 x 512
        x = torch.concat((vision, desire, traffic_convention), axis=1)
        x = self.elu(x)
        x = self._infeats0(x)
        x = self.relu(x)

        # initial state must have shape (1, N, hidden_dim) if batched
        # x must have shape (N, 1, input_dim) if batched
        rnn_out, _ = self._rnn(x.reshape(-1, 1, 1024), initial_state.reshape(1, -1, 512))
        rnn_out = rnn_out.squeeze(1) # squeeze rnn_out from (N, 1, 512) to (N, 512)
        out = torch.concat((rnn_out, x), axis=1)
        out = self._outfeats0(out)
        out = self.relu(out)

        return out, rnn_out

class SimpleResBlock(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.in_layer = nn.Linear(in_features=1024, out_features=hidden_dim, bias=True)
        self.res_layer1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.res_layer2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.final_layer = nn.Linear(hidden_dim, out_dim)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.relu(x)

        # send through residual layers
        y = self.res_layer1(x)
        y = self.relu(y)
        y = self.res_layer2(y)
        x = x + y

        # send through final layer and out
        x = self.relu(x)
        x = self.final_layer(x)
        return x

class FramePolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self._infeats0 = nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.meta = SimpleResBlock(64, 32)
        self.desire_pred = SimpleResBlock(32, 32)
        self.pose = SimpleResBlock(32, 12)

        self.elu = nn.ELU(alpha=1.0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # x is VisionNet output: 1 x 1024
        x = self.elu(x)
        x = self._infeats0(x)
        x = self.relu(x)

        meta = self.meta(x)
        desire_pred = self.desire_pred(x)
        pose = self.pose(x)

        return meta, desire_pred, pose

class TemporalPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.plan = SimpleResBlock(256, 4955)
        self.lead = SimpleResBlock(64, 255)
        self.lead_prob = SimpleResBlock(16, 3)
        self.lane_lines_prob = SimpleResBlock(16, 8)
        self.desire_state = SimpleResBlock(32, 8)
        self.lane_1_ll = SimpleResBlock(32, 132)
        self.lane_2_l = SimpleResBlock(32, 132)
        self.lane_3_r = SimpleResBlock(32, 132)
        self.lane_4_rr = SimpleResBlock(32, 132)
        self.lane_0_road_l = SimpleResBlock(16, 132)
        self.lane_5_road_r = SimpleResBlock(16, 132)
        
    def forward(self, x):
        # x is the output of temporal summarizer: 1 x 1024
        
        plan = self.plan(x)
        lead = self.lead(x)
        lead_prob = self.lead_prob(x)
        lane_lines_prob = self.lane_lines_prob(x)
        desire_state = self.desire_state(x)

        lane0 = self.lane_0_road_l(x).reshape(-1, 2, 66)
        lane1 = self.lane_1_ll(x).reshape(-1, 2, 66)
        lane2 = self.lane_2_l(x).reshape(-1, 2, 66)
        lane3 = self.lane_3_r(x).reshape(-1, 2, 66)
        lane4 = self.lane_4_rr(x).reshape(-1, 2, 66)
        lane5 = self.lane_5_road_r(x).reshape(-1, 2, 66)

        lane_group1 = torch.concat((lane1, lane2, lane3, lane4), axis=2).flatten(start_dim=1)
        lane_group2 = torch.concat((lane0, lane5), axis=2).flatten(start_dim=1)

        return plan, lead, lead_prob, lane_lines_prob, desire_state, lane_group1, lane_group2

class OpenPilotModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_net = VisionNet()
        self.temporal_summarizer = TemporalSummarizer()
        self.frame_policy = FramePolicy()
        self.temporal_policy = TemporalPolicy()

    def forward(self, input_imgs, desire, traffic_convention, recurrent_state):
        # send images through vision network
        vision = self.vision_net(input_imgs)

        ts_out, rnn_out = self.temporal_summarizer(vision, desire, traffic_convention, recurrent_state)
        meta, desire_pred, pose = self.frame_policy(vision)
        plan, lead, lead_prob, lane_lines_prob, desire_state, lane_group1, lane_group2 = self.temporal_policy(ts_out)

        # concat final outputs
        out = torch.concat((plan, lane_group1, lane_lines_prob, lane_group2, lead, lead_prob, desire_state, meta, desire_pred, pose, rnn_out), axis=1)
        return out
    

onnx_name_to_torch_name = {
    '1064': 'vision_net._bn0.running_var',
    '1065': 'vision_net.block0._bn1.running_var',
    '1066': 'vision_net.block0._bn2.running_var',
    '1067': 'vision_net.block1._bn1.running_var',
    '1068': 'vision_net.block1._bn2.running_var',
    '1069': 'vision_net.block2._bn0.running_var',
    '1070': 'vision_net.block2._bn1.running_var',
    '1071': 'vision_net.block2._bn2.running_var',
    '1072': 'vision_net.block3._bn0.running_var',
    '1073': 'vision_net.block3._bn1.running_var',
    '1074': 'vision_net.block3._bn2.running_var',
    '1075': 'vision_net.block4._bn0.running_var',
    '1076': 'vision_net.block4._bn1.running_var',
    '1077': 'vision_net.block4._bn2.running_var',
    '1078': 'vision_net.block5._bn0.running_var',
    '1079': 'vision_net.block5._bn1.running_var',
    '1080': 'vision_net.block5._bn2.running_var',
    '1081': 'vision_net.block6._bn0.running_var',
    '1082': 'vision_net.block6._bn1.running_var',
    '1083': 'vision_net.block6._bn2.running_var',
    '1084': 'vision_net.block7._bn0.running_var',
    '1085': 'vision_net.block7._bn1.running_var',
    '1086': 'vision_net.block7._bn2.running_var',
    '1087': 'vision_net.block8._bn0.running_var',
    '1088': 'vision_net.block8._bn1.running_var',
    '1089': 'vision_net.block8._bn2.running_var',
    '1090': 'vision_net.block9._bn0.running_var',
    '1091': 'vision_net.block9._bn1.running_var',
    '1092': 'vision_net.block9._bn2.running_var',
    '1093': 'vision_net.block10._bn0.running_var',
    '1094': 'vision_net.block10._bn1.running_var',
    '1095': 'vision_net.block10._bn2.running_var',
    '1096': 'vision_net.block11._bn0.running_var',
    '1097': 'vision_net.block11._bn1.running_var',
    '1098': 'vision_net.block11._bn2.running_var',
    '1099': 'vision_net.block12._bn0.running_var',
    '1100': 'vision_net.block12._bn1.running_var',
    '1101': 'vision_net.block12._bn2.running_var',
    '1102': 'vision_net.block13._bn0.running_var',
    '1103': 'vision_net.block13._bn1.running_var',
    '1104': 'vision_net.block13._bn2.running_var',
    '1105': 'vision_net.block14._bn0.running_var',
    '1106': 'vision_net.block14._bn1.running_var',
    '1107': 'vision_net.block14._bn2.running_var',
    '1108': 'vision_net.block15._bn0.running_var',
    '1109': 'vision_net.block15._bn1.running_var',
    '1110': 'vision_net.block15._bn2.running_var',
    '1111': 'vision_net.block16._bn0.running_var',
    '1112': 'vision_net.block16._bn1.running_var',
    '1113': 'vision_net.block16._bn2.running_var',
    '1114': 'vision_net.block17._bn0.running_var',
    '1115': 'vision_net.block17._bn1.running_var',
    '1116': 'vision_net.block17._bn2.running_var',
    '1117': 'vision_net.block18._bn0.running_var',
    '1118': 'vision_net.block18._bn1.running_var',
    '1119': 'vision_net.block18._bn2.running_var',
    '1120': 'vision_net.block19._bn0.running_var',
    '1121': 'vision_net.block19._bn1.running_var',
    '1122': 'vision_net.block19._bn2.running_var',
    '1123': 'vision_net.block20._bn0.running_var',
    '1124': 'vision_net.block20._bn1.running_var',
    '1125': 'vision_net.block20._bn2.running_var',
    '1126': 'vision_net.block21._bn0.running_var',
    '1127': 'vision_net.block21._bn1.running_var',
    '1128': 'vision_net.block21._bn2.running_var',
    '1129': 'vision_net.block22._bn0.running_var',
    '1130': 'vision_net.block22._bn1.running_var',
    '1131': 'vision_net.block22._bn2.running_var',
    '1132': 'vision_net._conv_head._bn1.running_var',
    '1133': 'vision_net._conv_head._bn2.running_var',
}

def load_weights_from_onnx(model):
    onnx_model = onnx.load('../openpilot_model/supercombo_server3.onnx')
    weights = onnx_model.graph.initializer

    # prepare map of ONNX weight names to PyTorch weight names
    # I recommend collapsing this for loop in your editor
    for i in range(len(weights)):
        onnx_name = weights[i].name
        torch_name = ''
        if 'frame_policy' in onnx_name:
            torch_name += 'frame_policy.'

            if 'desire_pred' in onnx_name:
                torch_name += 'desire_pred.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
            
            if 'meta' in onnx_name:
                torch_name += 'meta.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'

            if 'pose' in onnx_name:
                torch_name += 'pose.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
            
            if 'summarizer' in onnx_name:
                torch_name += '_infeats0.'

                if 'bias' in onnx_name:
                    torch_name += 'bias'
                elif 'weight' in onnx_name:
                    torch_name += 'weight'

        if 'temporal_policy' in onnx_name:
            torch_name += 'temporal_policy.'

            if 'plan' in onnx_name:
                torch_name += 'plan.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
            if 'lead.' in onnx_name:
                torch_name += 'lead.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
            
            if 'lead_prob' in onnx_name:
                torch_name += 'lead_prob.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'


            if 'lane_lines_prob' in onnx_name:
                torch_name += 'lane_lines_prob.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'


            if 'desire_state' in onnx_name:
                torch_name += 'desire_state.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '0' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '2' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'


            if 'lane_1_ll' in onnx_name:
                torch_name += 'lane_1_ll.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.0.' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.2.' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'


            if 'lane_2_l' in onnx_name:
                torch_name += 'lane_2_l.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.0.' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.2.' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'


            if 'lane_3_r' in onnx_name:
                torch_name += 'lane_3_r.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.0.' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.2.' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'


            if 'lane_4_rr' in onnx_name:
                torch_name += 'lane_4_rr.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.0.' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.2.' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'


            if 'lane_0_road_l' in onnx_name:
                torch_name += 'lane_0_road_l.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.0.' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.2.' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'

            
            if 'lane_5_road_r' in onnx_name:
                torch_name += 'lane_5_road_r.'

                if 'in_layer' in onnx_name:
                    torch_name += 'in_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.0.' in onnx_name:
                    torch_name += 'res_layer1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'res_layer' in onnx_name and '.2.' in onnx_name:
                    torch_name += 'res_layer2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                
                if 'final_layer' in onnx_name:
                    torch_name += 'final_layer.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'

        if 'temporal_summarizer' in onnx_name:
            torch_name += 'temporal_summarizer.'

            if '_infeats' in onnx_name:
                torch_name += '_infeats0.'

                if 'bias' in onnx_name:
                    torch_name += 'bias'
                elif 'weight' in onnx_name:
                    torch_name += 'weight'

            if '_outfeats' in onnx_name:
                torch_name += '_outfeats0.'

                if 'bias' in onnx_name:
                    torch_name += 'bias'
                elif 'weight' in onnx_name:
                    torch_name += 'weight'
            
            if '_rnn' in onnx_name:
                torch_name += '_rnn.'

                if 'x2h' in onnx_name and 'weight' in onnx_name:
                    torch_name += 'weight_ih_l0'
                if 'x2h' in onnx_name and 'bias' in onnx_name:
                    torch_name += 'bias_ih_l0'
                if 'h2h' in onnx_name and 'weight' in onnx_name:
                    torch_name += 'weight_hh_l0'
                if 'h2h' in onnx_name and 'bias' in onnx_name:
                    torch_name += 'bias_hh_l0'

        if 'vision' in onnx_name:
            torch_name += 'vision_net.'

            if 'blocks' in onnx_name:
                torch_name += 'block'
                block_num = onnx_name[19:21]
                if block_num[1] != '.':
                    block_num += '.'
                torch_name += block_num

                if '_depthwise_conv' in onnx_name:
                    torch_name += '_depthwise_conv.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'

                if '_project_conv' in onnx_name:
                    torch_name += '_project_conv.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'

                if '_expand_conv' in onnx_name:
                    torch_name += '_expand_conv.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'

                if '_bn0' in onnx_name:
                    torch_name += '_bn0.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                    elif 'running_mean' in onnx_name:
                        torch_name += 'running_mean'

                if '_bn1' in onnx_name:
                    torch_name += '_bn1.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                    elif 'running_mean' in onnx_name:
                        torch_name += 'running_mean'

                if '_bn2' in onnx_name:
                    torch_name += '_bn2.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                    elif 'running_mean' in onnx_name:
                        torch_name += 'running_mean'

            else:
                if '_bn0' in onnx_name:
                    torch_name += '_bn0.'

                    if 'bias' in onnx_name:
                        torch_name += 'bias'
                    elif 'weight' in onnx_name:
                        torch_name += 'weight'
                    elif 'running_mean' in onnx_name:
                        torch_name += 'running_mean'
                
                if '_conv_head' in onnx_name:
                    torch_name += '_conv_head.'

                    if '_depthwise_conv' in onnx_name:
                        torch_name += '_depthwise_conv.'
                        torch_name += 'weight'

                    if '_project_conv' in onnx_name:
                        torch_name += '_project_conv.'
                        torch_name += 'weight'

                    if '_expand_conv' in onnx_name:
                        torch_name += '_expand_conv.'
                        torch_name += 'weight'

                    if '_bn1' in onnx_name:
                        torch_name += '_bn1.'

                        if 'bias' in onnx_name:
                            torch_name += 'bias'
                        elif 'weight' in onnx_name:
                            torch_name += 'weight'
                        elif 'running_mean' in onnx_name:
                            torch_name += 'running_mean'

                    if '_bn2' in onnx_name:
                        torch_name += '_bn2.'

                        if 'bias' in onnx_name:
                            torch_name += 'bias'
                        elif 'weight' in onnx_name:
                            torch_name += 'weight'
                        elif 'running_mean' in onnx_name:
                            torch_name += 'running_mean'

                if '_conv_stem' in onnx_name:
                    torch_name += '_conv_stem.weight'

        if onnx_name not in onnx_name_to_torch_name:
            if torch_name == '':
                print('Empty:', onnx_name)
            onnx_name_to_torch_name[onnx_name] = torch_name

    model_weight_dict = {}

    for w in weights:
        torch_name = onnx_name_to_torch_name[w.name]
        model_weight_dict[torch_name] = torch.tensor(numpy_helper.to_array(w)).float()
    
    model.load_state_dict(model_weight_dict)
    # make sure to set the model state to evaluation so the BatchNorm layers don't change
    model.eval()

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    # load in preloaded example data
    input_imgs = np.load('test_inputs/input_imgs.npy')
    desire = np.load('test_inputs/desire.npy')
    traffic_convention = np.load('test_inputs/traffic_convention.npy')
    initial_state = np.load('test_inputs/initial_state.npy')

    # load ONNX model and pass input through
    session = onnxruntime.InferenceSession('../openpilot_model/supercombo_server3.onnx', None)
    onnx_res = session.run(['outputs'], {
        'input_imgs': input_imgs,
        'desire': desire,
        'traffic_convention': traffic_convention,
        'initial_state': initial_state
    })
    onnx_out = np.array(onnx_res[0])

    print("########## ONNX Model Output ##########")
    print(onnx_out.shape)
    print('Lead prob:', sigmoid(onnx_out[0, 6010:6013]))
    x_rel = onnx_out[0, 5755:5779:4]
    y_rel = onnx_out[0, 5755+1:5779:4]
    print('x_rel:', x_rel)
    print('y_rel:', y_rel)
    

    # load PyTorch model and pass input through
    model = OpenPilotModel()
    load_weights_from_onnx(model)
    torch_out = model(
        torch.tensor(input_imgs), 
        torch.tensor(desire), 
        torch.tensor(traffic_convention), 
        torch.tensor(initial_state)
    ).detach().numpy()

    
    print("########## PyTorch Model Output ##########")
    print(torch_out.shape)
    print('Lead prob:', sigmoid(torch_out[0, 6010:6013]))
    x_rel = torch_out[0, 5755:5779:4]
    y_rel = torch_out[0, 5755+1:5779:4]
    print('x_rel:', x_rel)
    print('y_rel:', y_rel)


    # save PyTorch model
    torch.save(model.state_dict(), '../openpilot_model/supercombo_torch_weights.pth')
    # torch.save(model, '../openpilot_model/supercombo_torch.pth')
