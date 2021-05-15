import numpy as np


def mixup_numpy(img1, img2, label1, label2)
    lambd = np.random.beta(1.5, 1.5)
    lambd = max(0, min(1, lambd))
    
    if lambd >= 1:
        weights1 = np.ones((label1.shape[0], 1))
        label1 = np.hstack((label1, weights1))
        return img1, label1
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = max(w1, w2)
    mix_img = np.zeros((h, w, 3), dtype = float)
    mix_img[:h1, :w1, :] = img1.asarray(dtype='float32')*lambd
    mix_img[:h2, :w2, :] += img2.asarray(dtype='float32')*(1-lambd)
    y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
    y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
    mix_label = np.vstack((y1, y2))
    return mix_img, mix_label