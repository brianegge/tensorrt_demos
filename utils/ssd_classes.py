"""ssd_classes.py

This file was modified from:
http://github.com/AastaNV/TRT_object_detection/blob/master/coco.py
"""

import os
import re

COCO_CLASSES_LIST = [
    'background',  # was 'unlabeled'
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

EGOHANDS_CLASSES_LIST = [
    'background',
    'hand',
]

def get_cls_dict(model):
    """Get the class ID to name translation dictionary."""
    label_file='ssd/' + model + '.txt'
    data_set=model.split('_')[-1]
    if model == 'coco':
        cls_list = COCO_CLASSES_LIST
    elif model == 'egohands':
        cls_list = EGOHANDS_CLASSES_LIST
    elif os.path.exists(label_file):
        with open(label_file) as fp:
            label_map={}
            for _, line in enumerate(fp):
                m = re.match(r'\s+id: ([0-9]+)', line)
                if m:
                    i = int(m[1])
                m = re.match('\s+name: \'(.*)\'', line)
                if m:
                    name = m[1]
                    label_map[i] = name
        print(label_map.items())
        return label_map
    else:
        raise ValueError('Bad model name {}'.format(model))
    return {i: n for i, n in enumerate(cls_list)}
