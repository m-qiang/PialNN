import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_config
from data.dataload import load_data, BrainDataset
from model.pialnn import PialNN
from utils import compute_normal, save_mesh_obj


if __name__ == '__main__':
    
    """set device"""
    if torch.cuda.is_available():
        device_name = "cuda:0"
    else:
        device_name = "cpu"
    device = torch.device(device_name)


    """load configuration"""
    config = load_config()
    
    
    """load data"""
    print("----------------------------")
    print("Start loading dataset ...")
    all_data = load_data(data_path=config.data_path,
                         hemisphere=config.hemisphere)

    L,W,H = all_data[0].volume[0].shape    # shape of MRI
    LWHmax = max([L,W,H])
    n_data = len(all_data)
    
    # split training / validation
    n_train = int(n_data * config.train_data_ratio)
    n_valid = n_data - n_train
    train_data = all_data[:n_train]
    valid_data = all_data[n_train:] 
    train_set = BrainDataset(train_data)
    valid_set = BrainDataset(valid_data)

    # batch size can only be 1
    trainloader = DataLoader(train_set, batch_size=1, shuffle=True)
    validloader = DataLoader(valid_set, batch_size=1, shuffle=False)
    
    print("Finish loading dataset. There are total {} subjects.".format(n_data))
    print("----------------------------")
    
    
    """load model"""
    print("Start loading model ...")
    model = PialNN(config.nc, config.K, config.n_scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.initialize(L, W, H, device)
    print("Finish loading model")
    print("----------------------------")

    
    """training"""
    print("Start training {} epochs ...".format(config.n_epoch))    
    for epoch in tqdm(range(config.n_epoch+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, v_gt, f_gt, v_in, f_in = data
            volume_in = volume_in.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            optimizer.zero_grad()

            v_out = model(v=v_in, f=f_in, volume=volume_in,
                          n_smooth=config.n_smooth, lambd=config.lambd)
            
            loss  = nn.MSELoss()(v_out, v_gt) * 1e+3
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        if config.report_training_loss:
            print("Epoch:{}, training loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % config.ckpts_interval == 0:
            print("----------------------------")
            print("Start validation ...")
            with torch.no_grad():
                error = []
                for idx, data in enumerate(validloader):
                    volume_in, v_gt, f_gt, v_in, f_in = data
                    volume_in = volume_in.to(device)
                    v_gt = v_gt.to(device)
                    f_gt = f_gt.to(device)
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)

                    v_out = model(v=v_in, f=f_in, volume=volume_in,
                                  n_smooth=config.n_smooth, lambd=config.lambd)
                    error.append(nn.MSELoss()(v_out, v_gt) * 1e+3)

            print("Validation error:{}".format(np.mean(avg_loss)))

            if config.save_model:
                print('Save model checkpoints ... ')
                path_save_model = "./ckpts/model/pialnn_model_"+config.hemisphere+"_"+str(epoch)+"epochs.pt"
                torch.save(model.state_dict(), path_save_model)

            if config.save_mesh_train:
                print('Save pial surface mesh ... ')
                path_save_mesh = "./ckpts/mesh/pialnn_mesh_"+config.hemisphere+"_"+str(epoch)+"epochs.obj"

                normal = compute_normal(v_out, f_in)
                v_gm = v_out[0].cpu().numpy() * LWHmax/2  + [L/2,W/2,H/2]
                f_gm = f_in[0].cpu().numpy()
                n_gm = normal[0].cpu().numpy()

                save_mesh_obj(v_gm, f_gm, n_gm, path_save_mesh)

            print("Finish validation.")
            print("----------------------------")

    print("Finish training.")
    print("----------------------------")