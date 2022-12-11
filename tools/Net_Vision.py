import cv2
import numpy as np

# 卷积层的可视化

def draw_cam1(x,orig_img, save_path=None):
   w,h,_ = orig_img.shape
   # x的shape为(W,H,channels)
   feature = x  # shape is w, keypoints

   feature = np.uint8(255*feature)
   heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)

   heatmap = cv2.resize(heatmap,(h,w))
   out = cv2.addWeighted(orig_img,0.6,heatmap,0.4,0)
   cv2.imshow("heatmap", out)
   cv2.waitKey(0)


