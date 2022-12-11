import math
import cv2
import time
from IPython import embed

def eval_distance(img, keypoints, dis_thres, count, step):
    x1, y1 = keypoints[8]
    x2, y2 = keypoints[10]

    x3, y3 = keypoints[11]
    x4, y4 = keypoints[13]
    if x1 or x2 != -1:
        distence = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
        if 0 < distence < dis_thres:  # 触发条件
            #cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            #cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
            count += 1
            if count > 0:
                fill_cnt = int(count * step)  # step是充电步长，数值越大触发时间越短
                '''
                    cv2.ellipse绘制椭圆，第二个参数【椭圆中心坐标】，(16, 16)长短轴，椭圆角度，0，0表示选择角度和椭圆起始角度,fill_cnt椭圆弧的终止角度，以度为单位
                    (255, 255, 0)是颜色，2是线的粗细
                    这个画椭圆是一帧一帧的画，直到画成一个完整的圆，即fil_cnt=360
                '''
                if fill_cnt < 360:  # 进入充电状态
                    cv2.ellipse(img, (int(x1), int(y1)), (16, 16), 0, 0, fill_cnt, (255, 255, 0), 2)
                    cv2.ellipse(img, (int(x2), int(y2)), (16, 16), 0, 0, fill_cnt, (255, 255, 0), 2)
                else:  # 充电完成
                    cv2.ellipse(img, (int(x1), int(y1)), (16, 16), 0, 0, fill_cnt, (0, 150, 255), 4)
                    cv2.ellipse(img, (int(x2), int(y2)), (16, 16), 0, 0, fill_cnt, (0, 150, 255), 4)

        else:  # 不满足触发条件
            count = 0
        return count


# 深蹲 {8, “RHip”},{9, “RKnee”}, {10, “RAnkle”},
#     {11, “LHip”},{12, “LKnee”},{13, “LAnkle”},

def vector_angle(img, keypoints, min_angle_thres, max_angle_thres, count_time, step):
    RHip_x, RHip_y = keypoints[8]
    RKnee_x, RKnee_y = keypoints[9]
    RAnkle_x, RAnkle_y = keypoints[10]

    LHip_x, LHip_y = keypoints[11]
    LKnee_x, LKnee_y = keypoints[12]
    LAnkle_x, LAnkle_y = keypoints[13]

    if RKnee_y or LKnee_y != -1:
        v1 = ((RHip_x - RKnee_x), (RHip_y - RKnee_y))
        v2 = ((RAnkle_x - RKnee_x), (RAnkle_y - RKnee_y))
        #cv2.arrowedLine(img, (RKnee_x, RKnee_y), (RHip_x, RHip_y), (0, 0, 255), thickness=2)

        v3 = ((LHip_x - LKnee_x), (LHip_y - LKnee_y))
        v4 = ((LAnkle_x - LKnee_x), (LAnkle_y - LKnee_y))

        v1_x, v1_y = v1
        v2_x, v2_y = v2

        v3_x, v3_y = v3
        v4_x, v4_y = v4

        v1v2 = v1_x * v2_x + v1_y*v2_y

        D_v1v2 = math.sqrt(math.pow(v1_x, 2) + math.pow(v1_y, 2)) * math.sqrt(math.pow(v2_x, 2) + math.pow(v2_y, 2))

        v3v4 = v3_x * v4_x + v3_y * v4_y
        D_v3v4 = math.sqrt(math.pow(v3_x, 2) + math.pow(v3_y, 2)) * math.sqrt(math.pow(v4_x, 2) + math.pow(v4_y, 2))

        try:
            angle_R = math.degrees(math.acos(v1v2 / (D_v1v2 + 1e-8)))
            angle_L = math.degrees(math.acos(v3v4 / (D_v3v4 + 1e-8)))

        except:
            angle_R = 65535
            angle_L = 65535
        if min_angle_thres < angle_R < max_angle_thres or min_angle_thres < angle_L < max_angle_thres:

            t1 = time.time()
            count_time += 1
            if count_time > 0:
                fill_cnt = int(count_time * step)  # step是充电步长，数值越大触发时间越短
                '''
                    cv2.ellipse绘制椭圆，第二个参数【椭圆中心坐标】，(16, 16)长短轴，椭圆角度，0，0表示选择角度和椭圆起始角度,fill_cnt椭圆弧的终止角度，以度为单位
                    (255, 255, 0)是颜色，2是线的粗细
                    这个画椭圆是一帧一帧的画，直到画成一个完整的圆，即fil_cnt=360
                '''
                if fill_cnt < 360:  # 进入充电状态
                    cv2.ellipse(img, (int(RKnee_x), int(RKnee_y)), (16, 16), 0, 0, fill_cnt, (255, 255, 0), 2)
                    cv2.ellipse(img, (int(LKnee_x), int(LKnee_y)), (16, 16), 0, 0, fill_cnt, (255, 255, 0), 2)

                else:  # 充电完成
                    cv2.ellipse(img, (int(RKnee_x), int(RKnee_y)), (16, 16), 0, 0, fill_cnt, (0, 150, 255), 4)
                    cv2.ellipse(img, (int(LKnee_x), int(LKnee_y)), (16, 16), 0, 0, fill_cnt, (0, 150, 255), 4)
        else:  # 不满足触发条件
            count_time = 0
        return count_time

