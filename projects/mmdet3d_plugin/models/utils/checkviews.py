from multiprocessing.spawn import import_main_path
from random import sample
import torch
import numpy as np
import cv2
from torchvision import transforms

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def tensor_to_image(samples):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    samp = samples.clone().cpu().squeeze(0)

    for i in range(len(mean)):
        samp[i] = samp[i].mul(std[i])+mean[i]

    img = transforms.ToPILImage()(samp)
    img = np.array(img)
    return img


def check(samples, bboxs, idx):

    # b, c, h, w = samples.shape
    save_path_1 = '/opt/data/private/jihao/Project/dab-streampetr-baseline/tools/cc_' + \
        str(idx)+'.jpg'

    img = tensor_to_image(samples)
    # samp = samples.clone().cpu().squeeze(0)
    # img = transforms.ToPILImage()(samp)
    # img = np.array(img)

    for bx in bboxs:
        # bx = box_cxcywh_to_xyxy(bx).tolist()
        x_1, y_1, x_2, y_2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        draw_2 = cv2.rectangle(img, (x_1, y_1),
                               (x_2, y_2), (0, 255, 0), 1)
    cv2.imwrite(save_path_1, draw_2)
