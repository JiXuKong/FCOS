# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2

# import libs.config as cfgs
PIXEL_MEAN = np.array([123.68, 116.779, 103.979])

# import libs.label_name_dict.coco_dict as coco_dict
# import libs.label_name_dict.label_dict as label_dict
# if cfgs.DATASET_NAME == 'coco':
#     LABEL_NAME_MAP = coco_dict.LABEL_NAME_MAP
# else:
#     LABEL_NAME_MAP = label_dict.LABEL_NAME_MAP
LABEL_NAME_MAP = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
    'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17,
    'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22,
    'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
    'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33,
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
    'kite': 38, 'baseball bat': 39, 'baseball glove': 40,
    'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
    'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48,
    'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53,
    'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57,
    'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61,
    'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
    'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
    'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77,
    'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81,
    'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86,
    'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90}

# LABEL_NAME_MAP = dict(zip(LABEL_NAME_MAP.values(), LABEL_NAME_MAP.keys()))
# print(LABEL_NAME_MAP['1'])
# print(LABEL_NAME_MAP)

LABEL_NAME_MAP = ['back_ground', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


# LABEL_NAME_MAP = dict(zip(np.arange(classes), classes))
# print(LABEL_NAME_MAP)

NOT_DRAW_BOXES = 0
ONLY_DRAW_BOXES = -1
ONLY_DRAW_BOXES_WITH_SCORES = -2

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'LightBlue', 'LightGreen'
]
FONT = ImageFont.load_default()


def draw_a_rectangel_in_img(draw_obj, box, color, width):
    '''
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    top_left, top_right = (x1, y1), (x2, y1)
    bottom_left, bottom_right = (x1, y2), (x2, y2)

    draw_obj.line(xy=[top_left, top_right],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[top_left, bottom_left],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[bottom_left, bottom_right],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[top_right, bottom_right],
                  fill=color,
                  width=width)


def only_draw_scores(draw_obj, box, score, color):

    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y-10, x+60, y],
                       fill=color)
    draw_obj.text(xy=(x, y),
                  text="obj:" +str(round(score, 2)),
                  fill='black',
                  font=FONT)


def draw_label_with_scores(draw_obj, box, label, score, color):
    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y-10, x + 60, y],
                       fill=color)
    # print(label, LABEL_NAME_MAP[label])
    # label = str(label)
    # quit()
    txt = LABEL_NAME_MAP[label] + ':' + str(round(score, 2))
    draw_obj.text(xy=(x, y-10),
                  text=txt,
                  fill='black',
                  font=FONT)


def draw_boxes_with_label_and_scores(img_array, boxes, labels, scores):

    img_array = img_array + np.array(PIXEL_MEAN)
    img_array.astype(np.float32)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    num_of_objs = 0
    for box, a_label, a_score in zip(boxes, labels, scores):

        if a_label != NOT_DRAW_BOXES:
            num_of_objs += 1
            draw_a_rectangel_in_img(draw_obj, box, color=STANDARD_COLORS[a_label], width=3)
            if a_label == ONLY_DRAW_BOXES:  # -1
                continue
            elif a_label == ONLY_DRAW_BOXES_WITH_SCORES:  # -2
                only_draw_scores(draw_obj, box, a_score, color='White')
                continue
            else:
                draw_label_with_scores(draw_obj, box, a_label, a_score, color='White')

    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.6)

    return np.array(out_img_obj)


# if __name__ == '__main__':
#     img_array = cv2.imread("/home/yjr/PycharmProjects/FPN_TF/tools/inference_image/2.jpg")
#     img_array = np.array(img_array, np.float32) - np.array(cfgs.PIXEL_MEAN)
#     boxes = np.array(
#         [[200, 200, 500, 500],
#          [300, 300, 400, 400],
#          [200, 200, 400, 400]]
#     )

#     # test only draw boxes
#     labes = np.ones(shape=[len(boxes), ], dtype=np.float32) * ONLY_DRAW_BOXES
#     scores = np.zeros_like(labes)
#     imm = draw_boxes_with_label_and_scores(img_array, boxes, labes ,scores)
#     # imm = np.array(imm)

#     cv2.imshow("te", imm)

#     # test only draw scores
#     labes = np.ones(shape=[len(boxes), ], dtype=np.float32) * ONLY_DRAW_BOXES_WITH_SCORES
#     scores = np.random.rand((len(boxes))) * 10
#     imm2 = draw_boxes_with_label_and_scores(img_array, boxes, labes, scores)

#     cv2.imshow("te2", imm2)
#     # test draw label and scores

#     labels = np.arange(1, 4)
#     imm3 = draw_boxes_with_label_and_scores(img_array, boxes, labels, scores)
#     cv2.imshow("te3", imm3)

#     cv2.waitKey(0)







