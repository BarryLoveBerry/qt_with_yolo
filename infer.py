'''
Author: huangwei barry.huangw@gmail.com
Date: 2024-12-20 15:01:18
LastEditors: huangwei barry.huangw@gmail.com
LastEditTime: 2024-12-24 10:59:22
FilePath: /pyqt6/yolov5/infer.py
Description: 推理流程
'''
import torch
import cv2
import numpy as np
import random


class Infer:
    def __init__(self, pt):
        self.model = torch.load(pt)
        self.label = ['crane','worker']
        self.color = [[255,0,0],[0,255,0],[0,0,255]]

    def infer_image(self, image):
        self.image = image.copy()
        self.results = self.model(image)
    
    def parse_result(self):
        detection_info = []
        if self.results is not None:
            for pred in self.results.pred[0]:
                x,y,w,h,conf,label = pred.tolist()
                detection_info.append(f"类别: {self.label[int(label)]}, 置信度: {conf:.2f}, 坐标: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})")
        return detection_info
    
    def box_iou(self, box1, box2):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[2:], box2[2:]) - torch.max(box1[:2], box2[:2])).clamp(0)
        inter = inter[0] * inter[1]
        return inter / (area1 + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def iou(self):
        flag = False
        temp_tensor = self.results.pred[0].clone()
        if self.results is not None:
            for i in range(len(temp_tensor.tolist()) - 1):
                for j in range(i+1, len(temp_tensor.tolist())):
                    overlap = self.box_iou(temp_tensor[i][...,:4],temp_tensor[j][...,:4])
                    if overlap > 0:
                        if int(temp_tensor[i][-1]) == 1 and int(temp_tensor[j][-1]) == 0:
                            temp_tensor[i][-1] = 2.0
                            flag = True
                        elif int(temp_tensor[i][-1]) == 0 and int(temp_tensor[j][-1]) == 1:
                            temp_tensor[i][-1] = 2.0
                            flag = True
        self.results.pred[0] = temp_tensor
        return flag

    def get_alert_info(self):
        return self.iou()

    def plot_pred(self):
        if self.results is not None:
            annotated_image = self.plot_box(self.image)
        return annotated_image
    
    def xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def plot_box(self,img,line_thickness=None):
        tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        for res in self.results.pred[0]:
            x1,y1,x2,y2,conf,class_name = res.tolist()
            c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))

            color = self.color[int(class_name)]
            label = self.label[(int(class_name) % len(self.label))]
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [255,255,255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        return img



        

