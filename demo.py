import argparse

import cv2
import numpy as np
import torch
from IPython import embed
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
from tools.Net_Vision import draw_cam1
from tools.Distance import eval_distance, vector_angle
#https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch#pre-trained-model

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height  # 图像缩放比例
    # fx和fy分别代表在width和height方向上的缩放比例  INTER_LINEAR双线性插值
    # cv2.resize(img, (0, 0), fx=scale, fy=scale 并不是将图像缩放到(0,0)大小，而是将图像按x轴和y轴分别缩放到scale大小
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]  # 得到热力图
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)  # shape is (128, 212, 19)

    stage2_pafs = stages_output[-1]  # 得到PAFs
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    count = 0
    count_time = 0
    upsample_ratio = 4

    num_keypoints = Pose.num_kpts # 18个关键点
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        #  heatmaps shape is (128, 212, 19) 19是每个通道是一个关键点，最后一个通道是背景类关键点   pafs shape (128, 212, 38)
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        #draw_cam1(heatmaps[:, :, 0], orig_img, r'./img/')

        total_keypoints_num = 0  # 关键点初始化
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            #draw_cam1(heatmaps[:, :, kpt_idx], orig_img, r'./img/')
            # 提取关键点 获得0~17关键点,这个total_keypoints_num会检测出当前帧或者当前图像中所有的点数
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)  # 获取关键点数量
        # all_keypoints_by_type列表长度为存放当前帧或者图像所有点的坐标信息，置信度和关键点序号(x,y,confidence,index)
        # all_keypoints是当前帧或图像所有关键点信息shape is (all_keyspoint,4)-->4维度(x,y,confidence,index)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        for kpt_id in range(all_keypoints.shape[0]):  # all_keypoints.shape[0]是所有的已检测出的关键点内遍历
            # 这一部分代码的意思是将得到的关键点映射到原图上对应的坐标点
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):  # 检测出来了几个人
            if len(pose_entries[n]) == 0:
                continue
            # pose_keypoints为什么是(18,2)的全-1矩阵呢？这是因为18指18个关键点，2指的是后面用来存放x,y的坐标信息
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1

            for kpt_id in range(num_keypoints):  # 在18关键点内遍历

                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            if args.Squat:
                count = eval_distance(img, pose_keypoints, args.dis_thres, count, args.step)  # 触发
                count_time = vector_angle(img, pose_keypoints, args.min_angle_thres, args.max_angle_thres, count_time, args.step)  # 深蹲检测

            # pose_keypoints中的关键点都是原图上对应的点
            pose = Pose(pose_keypoints, pose_entries[n][18])  # 把人框出来

            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)  # 绘制关键点
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        if args.video:
            out.write(img)
        else:
            delay = 0
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return


# python demo.py --checkpoint-path weights/checkpoint_iter_370000.pth --video 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height_size', type=int, default=256, help='network input layer height size')
    # parser.add_argument('--video', type=str, default='test4.mp4', help='path to video file or camera id')
    parser.add_argument('--video', type=str, default='test4.mp4', help='source')  # file/folder, 0 for webcam 0是打开摄像头
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', action='store_true',default=False, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--Squat', action='store_true', default=False, help='Squat detection')
    parser.add_argument('--dis_thres', type=int, default=160, help='keypoint distance')
    parser.add_argument('--min_angle_thres', type=float, default=45., help='keypoint min angle thres')
    parser.add_argument('--max_angle_thres', type=float, default=130., help='keypoint max angle thres')
    parser.add_argument('--step', type=int, default=40, help='power step')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
        for frame in frame_provider:
            size = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("test.avi", fourcc, 30, (size[1], size[0]))

    else:
        args.track = 0
    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)