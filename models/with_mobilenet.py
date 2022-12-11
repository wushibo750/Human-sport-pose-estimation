import torch
from torch import nn
from tools.Net_Vision import draw_cam1
from modules.conv import conv, conv_dw, conv_dw_no_bn


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),  # 该层为普通卷积 3*3*3*32 s=2
            conv_dw( 32,  64),  # 该层为深度可分离卷积，由一个3*3*32的深度卷积和一个1*1*32*64点卷积组成 s=1
            conv_dw( 64, 128, stride=2),  # 该层为深度可分离卷积，由一个3*3*64的深度卷积和一个1*1*64*128点卷积组成 s=2
            conv_dw(128, 128),  # 该层为深度可分离卷积，由一个3*3*128的深度卷积和一个1*1*128*128点卷积组成 s=1
            conv_dw(128, 256, stride=2),  # 该层为深度可分离卷积，由一个3*3*128的深度卷积和一个1*1*128*256点卷积组成 s=2
            conv_dw(256, 256),  # 该层为深度可分离卷积，由一个3*3*256的深度卷积和一个1*1*256*256点卷积组成 s=1
            conv_dw(256, 512),  # conv4_2  该层为深度可分离卷积，由一个3*3*256的深度卷积和一个1*1*256*512点卷积组成 s=1 在原mobilenet中s=2
            conv_dw(512, 512, dilation=2, padding=2),  # 加入了空洞卷积  由一个3*3*512的深度卷积和一个1*1*512*512点卷积组成
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        # 论文中在mobilenet出来后是cpm结构 cpm部分使用了残差结构，先进过1*1卷积进行通道的调整，这里是1*1*512*128，再经过一个3个3*3深度卷积(输出128通道)，最后再经过3*3卷积(输出128通道)
        # mobilenet和cpm是特征提取阶段
        self.cpm = Cpm(512, num_channels)
        # InitialStage进行heatmaps和pafs提取阶段
        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        # 经过cpm，initial_stage，出来的三个部分会进行一个cat拼接处理，然后进入一个stage，默认是1
        self.refinement_stages = nn.ModuleList() # 生成热力图
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)  # 生成热力图和pafs
        # draw_cam(stages_output[0], r'../img')
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[0], stages_output[1]], dim=1)))  # 按通道进行拼接

        return stages_output
