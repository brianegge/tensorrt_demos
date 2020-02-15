"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""


import sys
import time
import argparse
import os
import glob
import re

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'TrtSsdDemo'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [re.sub(r'.pb', '', i) for i in map(os.path.basename,glob.glob(os.path.join(os.path.dirname(__file__),'ssd','*.pb')))]
print(SUPPORTED_MODELS)
#[
#    'ssd_mobilenet_v1_coco',
#    'ssd_mobilenet_v1_egohands',
#    'ssd_mobilenet_v2_coco',
#    'ssd_mobilenet_v2_egohands',
#    'ssd_mobilenet_v2_garbage_bin',
#]


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='ssd_mobilenet_v2_coco',
                        choices=SUPPORTED_MODELS)
    parser.add_argument('--console', dest='use_console',
                        help='write output to console',
                        action='store_true')
    parser.add_argument('--once', dest='loop',
                        help='Run model once',
                        default=True,
                        action='store_false')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_ssd, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            boxes, confs, clss = trt_ssd.detect(img, conf_th)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

def loop_and_detect_console(cam, trt_ssd, conf_th, loop, cls_dict):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      loop: run continuously
    """
    fps = 0.0
    tic = time.time()
    run_loop=True # ensure we process the image once
    while run_loop:
        img = cam.read()
        if img is not None:
            boxes, confs, clss = trt_ssd.detect(img, conf_th)
            print(list(zip(list(map(cls_dict.get,clss)), confs, boxes)), end=" ")
            img = show_fps(img, fps)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            print('fps={}'.format(curr_fps))
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            run_loop = loop

def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    cls_dict = get_cls_dict(args.model)
    trt_ssd = TrtSSD(args.model, INPUT_HW)

    cam.start()
    if args.use_console:
        loop_and_detect_console(cam, trt_ssd, conf_th=0.3, loop=args.loop, cls_dict=cls_dict)
    else:
        open_window(WINDOW_NAME, args.image_width, args.image_height,
                    'Camera TensorRT SSD Demo for Jetson Nano')
        vis = BBoxVisualization(cls_dict)
        loop_and_detect(cam, trt_ssd, conf_th=0.3, vis=vis)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
