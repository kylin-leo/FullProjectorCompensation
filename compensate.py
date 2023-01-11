import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('GMA-main/core')
import torch
import torch.nn.functional as F
import numpy as np
from pronet import ProNet
import cv2
import torch.nn as nn



def imagewarp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W  = x.size()
    B1, H1, W1, C1 = flo.size()
    #print(B,C,H,W)
    # mesh grid
    xx = torch.arange(0, W1).view(1, -1).repeat(H1, 1)
    yy = torch.arange(0, H1).view(-1, 1).repeat(1, W1)
    xx = xx.view(1, H1, W1, 1).repeat(B1, 1, 1, 1)
    yy = yy.view(1, H1, W1, 1).repeat(B1, 1, 1, 1)
    grid = torch.cat((xx, yy), 3)

    x = x.cuda()
    grid = grid.cuda()
    vgrid =  flo + grid #Variable(grid)  # B,2,H,W

    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
    
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0  

    output = nn.functional.grid_sample(x.float(), vgrid.float(), mode='bilinear', padding_mode='zeros', align_corners=True)

    return output.permute(2, 3, 1, 0).view(H1*W1, C, 1)


m = 4
n1, n2 = 672, 896
num_sample = 70

version = 12

u = torch.from_numpy(np.load('flow_'+str(version)+'.npy')).float().cuda()

#cv2.namedWindow("proimg")
back_u = imagewarp(-u.permute(0, 3, 1, 2), -u).reshape(1, n1, n2, 2)

H = np.load('H_'+str(version)+'.npy')
H = torch.from_numpy(H).float().cuda()
back_H = imagewarp(H.view(672,896,m*3,1).permute(3,2,0,1), back_u).reshape(n1 * n2, m, 3)

promodel = ProNet(img_rows=672, img_cols=896, num_primary=m, num_filter=3 * k)
promodel.float().cuda()
promodel.load_state_dict(torch.load('./model/net_params_'+str(version)+'.pkl'))
promodel.eval()

for ii in range(num_sample):
    i = ii # (ii+1)%args.num_sample
    # i = math.floor(i/10)*10+1
    camimg = cv2.imread("dataset/target/target" + str(i) + ".jpg")
    camimg = camimg[:, :, ::-1] / 255.0

    print("camera image type:" + str(camimg.dtype))

    C = np.reshape(camimg,(672*896,3))

    C = torch.from_numpy(C).float().cuda()
    C = C.permute(1, 0).view(1, 3, 672, 896)

    back_C = imagewarp(C, back_u).reshape(n1 * n2, 3)

    P = back_C

    for j in range(40):

        outIm = promodel(P.view(672, 896, 3, 1).permute(3, 2, 0, 1)).permute(2,3,0,1)

        P = back_C - torch.matmul(outIm.view(672*896, 1, m),back_H).squeeze(1) + P

        # if i == 11:
        #     tmpP = torch.pow(P.view(672, 896, 3), 1 / 2.2)
        #     tmpP = 255 * tmpP[:, :, (2, 1, 0)].detach().cpu().numpy()
        #     cv2.imwrite("temp/P" + str(j) + ".jpg", tmpP[36:636, 48: 848, :])

    P = torch.pow(P.view(672, 896, 3), 1/2.2)
    P = 255 * P[:, :, (2, 1, 0)].detach().cpu().numpy()
    cv2.imwrite("comp/proimg" + str(i)  + "_"+str(version)+".png",  P[36:636, 48: 848, :])

