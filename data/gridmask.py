# import torch

# import torch.nn as nn

import numpy as np

from PIL import Image
import cv2
import pdb


def read_image(imgnm):
    return np.asarray(Image.open(imgnm))
    
def save_image(np_img, save_path):
    img = Image.fromarray(np_img, mode="RGB")
    img.save(save_path, quality=95)



class Grid(object):

    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.3, mode=0, prob = 1., use_object_drop = True):

        self.use_h = use_h

        self.use_w = use_w

        self.rotate = rotate

        self.offset = offset

        self.ratio = ratio

        self.mode=mode

        self.st_prob = prob

        self.prob = prob
        
        self.use_object_drop = use_object_drop



    def set_prob(self, epoch, max_epoch):

        self.prob = self.st_prob * epoch / max_epoch



    def __call__(self, img, label):

#         if np.random.rand() > self.prob:

#             return img, label

        h = img.shape[0]

        w = img.shape[1]

        self.d1 = 2

        self.d2 = min(h, w)

        hh = int(1.5*h)

        ww = int(1.5*w)
        d = np.random.randint(self.d1, 100)

#         d = np.random.randint(self.d1, self.d2)

        #d = self.d

#        self.l = int(d*self.ratio+0.5)

        if self.ratio == 1:

            self.l = np.random.randint(1, d)

        else:

            self.l = min(max(int(d*self.ratio+0.5),1),d-1)

#         self.l = np.random.randint(1, max(2, d//2))
        mask = np.ones((hh, ww), np.float32)

        st_h = np.random.randint(d)

        st_w = np.random.randint(d)

        if self.use_h:

            for i in range(hh//d):

                s = d*i + st_h

                t = min(s+self.l, hh)

                mask[s:t,:] *= 0

        if self.use_w:

            for i in range(ww//d):

                s = d*i + st_w

                t = min(s+self.l, ww)

                mask[:,s:t] *= 0

       

        r = np.random.randint(self.rotate)

        mask = Image.fromarray(np.uint8(mask))

        mask = mask.rotate(r)

        mask = np.asarray(mask)

#        mask = 1*(np.random.randint(0,3,[hh,ww])>0)

        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
#         cv2.imshow("mask",mask)
#         cv2.waitKey(0)
        #对目标进行去mask
        mask1 = np.zeros((h, w), np.float32)
        for i in range(len(label)):
            x1,y1,x2,y2 = label[i][0], label[i][1], label[i][2], label[i][3]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            box_scale = min(int(x2-x1),int(y2-y1))
            if self.l>=box_scale//2:
                mask1[y1:y2, x1:x2] = 1
                drop_point = np.random.randint(5, 10)
                for j in range(drop_point):
                    dropx = np.random.randint(x1,x2-1)
                    dropy = np.random.randint(y1,y2-1)
                    if ((x2-dropx)//2>0)and((y2-dropy)//2>0):
                        drop_lenx = np.random.randint((x2-dropx)//2)
                        drop_leny = np.random.randint((y2-dropy)//2)
                        obj_drop_x1, obj_drop_y1 = dropx, dropy
                        obj_drop_x2, obj_drop_y2 = dropx+drop_lenx, dropy+drop_leny
                        mask1[obj_drop_y1:obj_drop_y2, obj_drop_x1:obj_drop_x2] = 0
                
         
        mask1 = Image.fromarray(np.uint8(mask1))
        mask1 = np.asarray(mask1)
        mask2 = mask1 + mask
        for i in range(mask2.shape[0]):
            for j in range(mask2.shape[1]):
                if mask2[i,j] == 2:
                    mask2[i,j] = 1
                
        
#         mask = torch.from_numpy(mask).float()

        if self.mode == 1:

            mask = 1-mask



        mask = np.reshape(mask, (h,w,1))

        mask2= np.reshape(mask2, (h,w,1))

        if self.use_object_drop:
            mask = mask2
        else:
            mask = mask

        if self.offset:

#             offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float()

            offset = 0
            offset = (1 - mask) * offset

            img = img * mask + offset

        else:

            img = img * mask



        return img, label, mask

