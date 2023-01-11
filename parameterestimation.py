# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from matplotlib import pyplot as plt
import torch
from skimage.color import rgb2gray
import cv2

from sklearn.linear_model import OrthogonalMatchingPursuit

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('RAFT-master/core')
import torch
import torch.nn.functional as F
from pronet import ProNet
import argparse
from imagewarp import imagewarp
import numpy as np
from raft import RAFT
import cv2
import torch.nn as nn
import time



num_sample = 12
k = 4
out_iter = 400

tau = 0.03
eta = 0.5
alpha = 0.1
beta = 0.1
m = 4

def imagewarp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    B1, H1, W1, C1 = flo.size()
    # print(B,C,H,W)
    # mesh grid
    xx = torch.arange(0, W1).view(1, -1).repeat(H1, 1)
    yy = torch.arange(0, H1).view(-1, 1).repeat(1, W1)
    xx = xx.view(1, H1, W1, 1).repeat(B1, 1, 1, 1)
    yy = yy.view(1, H1, W1, 1).repeat(B1, 1, 1, 1)
    grid = torch.cat((xx, yy), 3)

    x = x.cuda()
    grid = grid.cuda()
    vgrid = flo + grid  # Variable(grid)  # B,2,H,W
    
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
   
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0  

    
    output = nn.functional.grid_sample(x.float().cuda(), vgrid.float().cuda(), mode='bilinear', padding_mode='zeros',
                                       align_corners=True)

    return output.permute(2, 3, 1, 0).view(H1 * W1, C, 1)

def opt_u(u0, proimg, Wimg, model):

    num_sample, size_x, size_y, num_channel = proimg.shape

    u = torch.zeros((672, 896, 2)).cuda()
    model.eval()
    j = 0

    for i in range(num_sample):
        tmp2 = proimg[i, :, :, :].unsqueeze(0).permute(0, 3, 1, 2)
        tmp1 = Wimg[i, :, :, :].unsqueeze(0).permute(0, 3, 1, 2)
        tmp1 = tmp1 * torch.sum(tmp2) / torch.sum(tmp1)
        #tmp2 = torch.from_numpy(tmp2).float().cuda()
        #tmp1 = torch.from_numpy(tmp1).float().cuda()
        _, newu = model(tmp1[:,0:3,:,:].cuda(), tmp2[:,0:3,:,:].cuda(), iters=12, test_mode=True)
        newu = newu.detach().permute(0, 2, 3, 1)

        if torch.max(torch.abs(newu)) > 250:
            continue
        else:
            diff0 = imagewarp(tmp2.cuda(), u0).view(672, 896, 1, m).permute(2, 3, 0, 1) - tmp1.cuda()
            diffnew = imagewarp(tmp2.cuda(), newu).view(672, 896, 1, m).permute(2, 3, 0, 1) - tmp1.cuda()

            if torch.mean(diff0) > torch.mean(diffnew):

                u = (u * j + newu)/(j + 1)
                j = j + 1
    print("j=",j)

    if j == 0:
        return u0
    else:
        return u

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default = "RAFT-master/raft-sintel.pth")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    promodel = ProNet(img_rows=672, img_cols=896, num_primary=m, num_filter=3 * k)
    promodel.float()
    promodel.cuda()
    # promodel.train()
    prooptimizer = torch.optim.Adam(promodel.parameters(), lr=3e-5)
    #gma_model.cuda()

    proimg = np.zeros((num_sample, 672, 896, 3))
    camimg = np.zeros((num_sample, 672, 896, 3))

    cv2.namedWindow("proimg")
    cv2.namedWindow("camimg")

    for i in range(num_sample):

        tmpcamimg = cv2.imread("dataset/captured/img"+str(i+1)+".tiff") #- cv2.imread("dataset/captured/img78.tiff")
        camimg[i,:,:,:] = cv2.resize(tmpcamimg[:, :, ::-1]/255.0, (896, 672))

        tmpproimg = cv2.imread("dataset/projected/seq"+str(i+0)+".jpg")
        proimg[i, 36:636, 48:848, :] = np.power(tmpproimg[:, :, ::-1]/255.0, 2.2)


# % W -- (num_sample x 672 x 896) x m
# % A -- k x (num_sample x 672 x 896)
# % C -- (num_sample x 672 x 896) x 3

    proimg = torch.from_numpy(proimg).float().cuda()
    u = torch.zeros((1, 672, 896, 2)).float().cuda()
    W = torch.from_numpy(np.ones((672 * 896, num_sample, m))).float().cuda()

    C0 = torch.from_numpy(np.transpose(np.reshape(camimg, (num_sample, 672 * 896, 3)), (1, 0, 2))).float().cuda()
    # n x num x 3

    warpI = torch.ones((num_sample, 672, 896, m)).cuda()

    H = torch.ones(672 * 896, m, 3).cuda()
    lrH = H


    tensor_proimg = proimg.permute(0, 3, 1, 2)

    out_Im = torch.zeros(num_sample, m, 672, 896)
    out_Im[:, 0:3, :,:] = tensor_proimg

    warpI = imagewarp(out_Im.reshape(1, num_sample*m, 672, 896), u)
    warpI = warpI.view(672*896, num_sample, m)

    for i in range(out_iter):
        print(i)

        time_start = time.time()

        gW = torch.matmul(torch.matmul(W, H) - C0, H.permute(0, 2, 1)) +  tau * (W - warpI)
        W = W - alpha * gW

        time_end = time.time()
        print('time cost-W', time_end - time_start, 's')


        time_start = time.time()

        gH = torch.matmul(W.permute(0, 2, 1), torch.matmul(W, H) - C0) + eta * (H - lrH)
        H = H - beta * gH
        time_end = time.time()
        print('time cost-H', time_end - time_start, 's')


        time_start= time.time()

        promodel.train()
        #pe = positionencoding3D(tensor_proimg, 2)
        out_Im = promodel(tensor_proimg)#promodel(pe).cuda()
        #
        warpI = imagewarp(out_Im.reshape(1, num_sample*m, 672, 896), u).reshape(672*896, num_sample, m)
        #
        loss_mse = nn.MSELoss()(W.detach(), warpI)

        prooptimizer.zero_grad()
        loss_mse.backward()
        prooptimizer.step()

        time_end = time.time()
        print('time cost-theta', time_end - time_start, 's')

        # res = torch.isnan(H)
        #
        # if torch.sum(res):
        #     print(res)

        time_start = time.time()

        warpI = warpI.detach()

        [uh, sh, vh] = torch.linalg.svd(H.view(672*896, m*3), full_matrices= False)

        sh[k:] = 0
        lrH = torch.matmul(torch.matmul(uh, torch.diag(sh)),vh)
        lrH = lrH.reshape(672*896, m, 3)

        #print(vm)
        time_end = time.time()
        print('time cost-Hprime', time_end - time_start, 's')

        Wimg = W.permute(1, 0, 2).view(num_sample, 672, 896, m)

        time_start = time.time()
        if i < 4:
            u = opt_u(u, out_Im.permute(0,2,3,1), Wimg, model)
        time_end = time.time()
        print('time cost-u', time_end - time_start, 's')

        if np.mod(i, 10) == 0:

            D = (torch.matmul(W, H) - C0).cpu().detach().numpy()
            D = np.transpose(D, (1, 0, 2))

            Aimg = torch.matmul(uh, torch.diag(sh)).view(672, 896, 3*m)

            #warpI = warpI.detach().cpu().numpy()
            Aimg = Aimg.cpu().detach().numpy()
            D = np.reshape(D, (num_sample, 672, 896, 3))
            Wimg = Wimg.cpu().detach().numpy()
            warpImg = warpI.permute(1, 0, 2).reshape(num_sample, 672, 896, m).detach()

            multi_img = np.hstack([warpImg[11,:,:,0:3].cpu().numpy(), Wimg[11,:,:,0:3], 2*camimg[11,:,:,:], 50.0*D[11,:,:,0:3], -50.0*D[11,:,:, 0:3]])
            cv2.imshow("proimg", cv2.resize(multi_img[:,:,::-1], (int(896*5/4),int(672/4))))
            cv2.imwrite("temp/tempWnD"+str(i)+".jpg", 255.0*cv2.resize(multi_img[:,:,::-1], (int(896*5/2),int(672/2))))
            cv2.waitKey(100)
            multi_img2 = np.hstack([Aimg[:,:,0], Aimg[:,:,1], Aimg[:,:,2], Aimg[:,:,3]])
            multi_img3 = np.hstack([-Aimg[:, :, 0], -Aimg[:, :, 1], -Aimg[:, :, 2], -Aimg[:, :, 3]])
            multi_img = np.vstack([multi_img2, multi_img3])
            cv2.imshow("camimg", cv2.resize(multi_img, (int(896*2/4),int(672/4))))
            cv2.imwrite("temp/tempH"+str(i)+".jpg", 255.0*cv2.resize(multi_img, (int(896*2),int(672))))
            cv2.waitKey(100)

        tau = min(0.5, tau * 1.1)



    np.save('H_'+str(num_sample)+'.npy', H.detach().cpu().numpy())
    u = u.detach().cpu().numpy()
    np.save('flow_'+str(num_sample)+'.npy', u)

    torch.save(promodel.state_dict(), "./model/net_params_"+str(num_sample)+".pkl")

