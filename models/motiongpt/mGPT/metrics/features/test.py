import sys
sys.path.append('/mnt/AFS_jiangjianping/projects/SEA/models/motiongpt/mGPT/metrics/features')
import numpy as np
import pickle 
from kinetic import extract_kinetic_features as extract_kinetic_features_np
from kinetic import KineticFeatures
from scipy import linalg
import os 
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import json
import torch
import time
data1 = np.random.rand(64, 140, 52, 3)
m_seqs = [130] * 64
# a = extract_kinetic_features_np(data1)



np_time_start = time.time()
res_h_np = []
for j in range(data1.shape[0]):
    features_np = KineticFeatures(data1[j][:m_seqs[j]])
    kinetic_feature_vector = []
    for i in range(data1[j].shape[1]):
        feature_vector = np.array(
            [   features_np.average_kinetic_energy_horizontal(i),
                features_np.average_kinetic_energy_vertical(i),
                features_np.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32)
    res_h_np.append(kinetic_feature_vector)

res_h_np = np.array(res_h_np)

print('Origin ')
print(time.time() - np_time_start)


def cal_average_kinetic_energy_torch(data, m_lens, frame_time=1./30, up_vec="y", sliding_window=2):
    if type(data) == np.ndarray:
        data = torch.tensor(data)

    B, L, J, _ = data.shape    
    # assert 2* sliding_window + 1 < seq_len
    mask = torch.zeros(B, L, J, 1).to(data.device)
    for i, m_len in enumerate(m_lens):
        mask[i, :m_len] = 1
    velocity = (data[:, 1:] - data[:, :-1]) / frame_time
    velocity = velocity * mask[:, 1:]
    # padding 0 and conv to get the mean velocity
    window_size = 2 * sliding_window + 1
    
    velocity_padded = torch.nn.functional.pad(velocity, (0, 0, 0, 0, sliding_window, sliding_window), mode='constant', value=0)
    mask_padded = torch.nn.functional.pad(mask[:, 1:], (0, 0, 0, 0, sliding_window, sliding_window), mode='constant', value=0)
    velocity_unfolded = velocity_padded.unfold(1, window_size, 1)
    mask_unfolded = mask_padded.unfold(1, window_size, 1)
    velocity_sum = velocity_unfolded.sum(dim=-1)
    velocity_sum = velocity_sum * mask[:, 1:]
    mask_sum = mask_unfolded.sum(dim=-1)
    mask_sum_tag  = torch.where(mask_sum == 0, 1e-7, mask_sum)
    velocity_mean = velocity_sum / mask_sum_tag
    
    if up_vec == "y":
        velocity_mean_h = velocity_mean[:, :, :, [0, 2]]
        energy_h = velocity_mean_h.pow(2).sum(dim=-1).sum(dim=1)
        mask_len = torch.tensor([m_len-1 for m_len in m_lens]).to(data.device)
        mask_len = mask_len.unsqueeze(-1)
        energy_h_final = energy_h / mask_len
        
        velocity_mean_v = velocity_mean[:, :, :, 1]
        energy_v = velocity_mean_v.pow(2).sum(dim=1)
        energy_v_final = energy_v / mask_len
        # return energy_h_final, energy_v_final
    elif up_vec == "z":
        velocity_mean_h = velocity_mean[:, :, :, [0, 1]]
        energy_h = velocity_mean_h.pow(2).sum(dim=-1).sum(dim=1)
        mask_len = torch.tensor([m_len-1 for m_len in m_lens]).to(data.device)
        mask_len = mask_len.unsqueeze(-1)
        energy_h_final = energy_h / mask_len
        
        velocity_mean_v = velocity_mean[:, :, :, 2]
        energy_v = velocity_mean_v.pow(2).sum(dim=1)
        energy_v_final = energy_v / mask_len
        # return energy_h_final, energy_v_final
    else:
        raise NotImplementedError
    
    acc = (velocity[:, 1:] - velocity[:, :-1]) / frame_time
    acc = acc * mask[:, 2:]
    acc_padded = torch.nn.functional.pad(acc, (0, 0, 0, 0, sliding_window, sliding_window+1), mode='constant', value=0)
    mask_padded_acc = torch.nn.functional.pad(mask[:, 2:], (0, 0, 0, 0, sliding_window, sliding_window+1), mode='constant', value=0)
    acc_unfolded = acc_padded.unfold(1, window_size, 1)
    mask_unfolded_acc = mask_padded_acc.unfold(1, window_size, 1)
    acc_sum = acc_unfolded.sum(dim=-1)
    acc_sum = acc_sum * mask[:, 1:]
    mask_sum_acc = mask_unfolded_acc.sum(dim=-1)
    mask_sum_tag_acc  = torch.where(mask_sum_acc == 0, 1e-7, mask_sum_acc)
    acc_mean = acc_sum / mask_sum_tag_acc
    energy_acc = torch.norm(acc_mean, p=2, dim=-1)
    energy_acc = energy_acc.sum(dim=1)
    energy_acc_final = energy_acc / mask_len
    
    return energy_h_final, energy_v_final, energy_acc_final
pass


data_torch = torch.tensor(data1).to('cuda')
print('Torch ')
torch_time_start = time.time()
res_h, res_v, res_acc = cal_average_kinetic_energy_torch(data_torch, m_seqs)

stacked = torch.stack((res_h, res_v, res_acc), dim=-1)  # 结果维度为 (B, L, 3)
B, L, _ = stacked.shape
interleaved = stacked.reshape(B, L*3)

print(time.time() - torch_time_start)
print('Max error')
print(np.abs((interleaved.cpu().numpy() - res_h_np)).max())
pass