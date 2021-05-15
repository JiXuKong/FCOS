class_num = 21
fpn_channel = 256

strides = [8, 16, 32, 64, 128]
limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
score_threshold=0.05
nms_iou_threshold=0.6
max_detection_boxes_num = 1000