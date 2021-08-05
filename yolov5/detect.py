import argparse
import time

import cv2
import numpy as np
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, colorstr, non_max_suppression, scale_coords
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized


class YoloDetector:
    def __init__(self,weights='yolov5s.pt',input_size=640,device=''):
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.input_size = check_img_size(input_size, s=self.stride)  # check image size

        self.cls_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, input_size, input_size).to(device).type_as(next(self.model.parameters())))  # run once

    def __preprocess(self, filepath):
        img_ori = cv2.imread(filepath)  # BGR
        assert img_ori is not None, 'Image Not Found ' + filepath

        # Padded resize
        img = letterbox(img_ori, self.input_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_ori,img

    def __postprocess(self, preds, ori_shape, resize_shape):
        # Process predictions
        results = []
        for i,pred in enumerate(preds):  # detections per image
            box = pred[:,:4]
            box = scale_coords(resize_shape,box,ori_shape).round().numpy()
            cls = pred[:,5:6].numpy()
            conf = pred[:, 4:5].numpy()
            results.append(np.concatenate([cls,conf,box],axis=1))
        return results

    @torch.no_grad()
    def detect(self,
               filepath,
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               show_result=False
               ):

        t0 = time.time()

        img_ori,img = self.__preprocess(filepath)
        ori_shape = img_ori.shape[:2]
        resize_shape = img.shape[1:]
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0


        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_synchronized()
        preds = self.model(img)[0]

        # NMS
        preds= non_max_suppression(preds, conf_thres, iou_thres, None, False, max_det=max_det)

        results = self.__postprocess(preds,ori_shape,resize_shape)

        t2 = time_synchronized()

        # Print time (inference + NMS)
        print(f'Done. ({t2 - t1:.3f}s)')

        if show_result:
            for result in results:
                for cls, conf, *box in reversed(result):
                    c = int(cls)  # integer class
                    label = (f'{self.cls_names[c]} {conf:.2f}')
                    plot_one_box(box, img_ori, label=label, color=colors(c, True), line_thickness=3)

                # Stream results
                cv2.imshow(filepath, img_ori)
                cv2.waitKey(0)


        print(f'Done. ({time.time() - t0:.3f}s)')

        return results




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='file path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--input_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    D = YoloDetector(opt.weights,opt.input_size,opt.device)
    pred = D.detect(opt.file_path,opt.conf_thres,opt.iou_thres,opt.max_det,show_result=True)
    print(pred)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)