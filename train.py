from Model.model import DehazeNetwork, ContrastLoss
from data import DenseHaze_test_loader, DenseHaze_train_loader
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import time
from metric import ssim, psnr 
from utils import plot, multiPlot
import torchvision.transforms as T
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--cr_coeff', type=float, default=0.1, help='number of workers')
    start = time.time()
    opt = parser.parse_args()

    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    train_data = DenseHaze_train_loader
    test_data = DenseHaze_test_loader
    model = DehazeNetwork(3,3).to(device)

    transform = T.ToPILImage()

    L1Loss = nn.L1Loss().to(device)
    cr_loss = ContrastLoss(ablation=False).to(device)
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()

    l1_loss_list = []
    cr_loss_list = []
    total_lost_list = []

    print("Training Started")
    total_step = len(train_data)

    psnr_loss_tr_sum, ssim_loss_tr_sum = 0, 0
    psnr_loss_te_sum, ssim_loss_te_sum = 0, 0
    psnr_tr_list, ssim_tr_list = [], []
    psnr_te_list, ssim_te_list = [], []

    for i in range(opt.epochs):
        l1_loss_curr = 0
        cr_loss_curr = 0
        total_loss_curr = 0
        model.train()

        for j in range(len(train_data)):
            

            iterator = iter(train_data)
            gt, hazy = next(iterator)
            gt = gt.to(device)
            hazy = hazy.to(device)

            out = model(hazy)
            
            #compute l1 loss and cr loss
            l1 = L1Loss(out, gt)
            cr_los = cr_loss(out, gt, hazy)

            loss = l1 + opt.cr_coeff * cr_los

            total_loss_curr += loss.item()
            l1_loss_curr += l1.item()
            cr_loss_curr += cr_los.item()

            ssim_tr = ssim(out.detach(), gt.detach())
            psnr_tr = psnr(out.detach(), gt.detach())
            psnr_loss_tr_sum += psnr_tr
            ssim_loss_tr_sum += ssim_tr.cpu().numpy()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch [{i + 1}/{opt.epochs}]')
        print(f'[Train] L1 Loss : {l1_loss_curr / total_step:.5f}, CR Loss : {cr_loss_curr / total_step:.5f}, Total Loss : {total_loss_curr / total_step:.5f}')


        #append result to use for plotting
        total_lost_list.append(total_loss_curr / total_step)
        l1_loss_list.append(l1_loss_curr / total_step)
        cr_loss_list.append(cr_loss_curr / total_step)
        psnr_tr_list.append(psnr_loss_tr_sum / total_step)
        ssim_tr_list.append(ssim_loss_tr_sum / total_step)


        #test
        with torch.no_grad():

            torch.cuda.empty_cache()
            ssims = []
            psnrs = []
            s_total = 0
            p_total = 0

            for i, (targets, inputs) in enumerate(test_data):
                inputs = inputs.to(device);targets = targets.to(device)
                with torch.no_grad():
                    pred = model(inputs)

                ssim1 = ssim(pred, targets)
                s_total += ssim1.cpu().numpy()
                psnr1 = psnr(pred, targets)
                p_total += psnr1
                ssims.append(ssim1)
                psnrs.append(psnr1)


                print('Ssim: {}, Psnr: {}'.format(ssim1, psnr1))
            
            print('Average Ssim: {}, Average Psnr: {}'.format(s_total / len(test_data), p_total / len(test_data)))
            psnr_te_list.append(p_total / len(test_data))
            ssim_te_list.append(s_total / len(test_data))
            


    #save loss plots
    plot("L1", "L1_LOSS" + str(opt.epochs) + "_" + str(opt.lr) + "_" + str(opt.cr_coeff), l1_loss_list)
    plot("CR", "CR_LOSS" + str(opt.epochs) + "_" + str(opt.lr) + "_" + str(opt.cr_coeff), cr_loss_list)
    plot("Total", "Total_LOSS" + str(opt.epochs) + "_" + str(opt.lr) + "_" + str(opt.cr_coeff), total_lost_list)

    #save model
    #torch.save(model.state_dict(), './models/' + str(opt.epochs) + "_" + str(opt.lr) + "_" + str(opt.cr_coeff) + '.pth')
    
    end = time.time()
    print("Training Time: ", end - start)
    

    del model

    print("End of the experiment")


if __name__ == "__main__":
       main()


