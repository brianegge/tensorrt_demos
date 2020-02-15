#!/bin/sh

# ./download_yolov3.sh
python3 yolov3_to_onnx.py --model ${1:-yolov3-416}
python3 onnx_to_tensorrt.py --model ${1:-yolov3-416}
