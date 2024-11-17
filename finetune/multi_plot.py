import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from os import path as osp
from pdb import set_trace as stx
import numpy as np


# def smooth(data, window_size=50):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def smooth(y, alpha):
    res = y
    # alpha = 0.01
    for i in range(1, len(y)):
        res[i] = alpha * y[i] + (1 - alpha) * res[i - 1]
    return res


def load_train_data(dir):
    path = osp.join(dir, 'train_data.json')
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data    


def plot_multi_curve(rks, output_dir):
    paths = [osp.join('results', f'_rk{r}') for r in rks]
    datas = [load_train_data(path) for path in paths] 
    data_dict = {}
    keys = ['train_steps', 'train_loss', 'eval_steps', 'eval_loss', ]
    for key in keys:
        data_dict[key+'_lst'] = [data[key] for data in datas]

    for split in ['train', 'eval']:
        steps_lst = data_dict[f'{split}_steps_lst']
        loss_lst = data_dict[f'{split}_loss_lst']
        # if split == 'train':
        plt.figure(figsize=(12, 6))
        for r, steps, loss in zip(rks, steps_lst, loss_lst):
            alpha = 0.01 if split == 'train' else 0.5
            loss = smooth(loss, alpha)
            plt.plot(steps, loss, label=f'rank {r}', alpha=1, linewidth=0.5)
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.title(split)
        plt.legend()
        plt.savefig(osp.join(output_dir, f"{split}.png"))
        plt.close()


def plot_single(dir, output_dir):
    train_data = load_train_data(dir)
    train_steps = train_data["train_steps"]
    train_loss = train_data["train_loss"]
    train_loss = smooth(train_loss, alpha=0.01)
    eval_steps = train_data["eval_steps"]
    eval_loss = train_data["eval_loss"]
    eval_loss = smooth(eval_loss, alpha=0.5)
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_loss, label="train", alpha=1, linewidth=0.5)
    plt.plot(eval_steps, eval_loss, label="eval", alpha=1, linewidth=0.5)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(osp.join(output_dir, "rk32.png"))
    plt.close()


if __name__ == '__main__':
    # rks = [1, 2, 4, 8, 16, 32]
    # plot_multi_curve(rks, 'imgs')
    
    # plot_single('results/_rk32', 'imgs')
    plot_single('results/_rk32_B16_sca16', 'imgs')
    
    # settings = ['32_B8_sca32', '32_B16_sca16', '32_B16_sca32']
    # plot_multi_curve(settings, 'imgs')