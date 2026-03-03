from utils.tools import *
import itertools
from scipy.linalg import hadamard
from network import *
import pdb
import os

import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from losses import *

from collections import defaultdict

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--hash_dim", type=int, default=32)
parser.add_argument("--noise_rate", type=float, default=1.0)
parser.add_argument("--dataset", type=str, default="flickr")
parser.add_argument("--Lambda", type=float, default=0.6)
parser.add_argument("--num_gradual", type=int, default=100)
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
Lambda = args.Lambda
num_gradual = args.num_gradual

if dataset == "flickr":
    train_size = 10000
elif dataset == "ms-coco":
    train_size = 10000
elif dataset == "nuswide21":
    train_size = 10500
elif dataset == "iapr":
    train_size = 10000
n_class = 0
tag_len = 0
# # torch.multiprocessing.set_sharing_strategy("file_system")


def get_config():
    config = {
        "optimizer": {
            "type": optim.RMSprop,
            "optim_params": {"lr": 1e-5, "weight_decay": 10**-5},
        },
        "txt_optimizer": {
            "type": optim.RMSprop,
            "optim_params": {"lr": 1e-5, "weight_decay": 10**-5},
        },
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 256,
        "dataset": dataset,
        "epoch": 100,
        "device": torch.device("cuda:"+args.gpus),
        "bit_len": bit_len,
        "noise_type": "symmetric",
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "lambda": Lambda,
        "tag_len": tag_len,
        "train_size": train_size,
        "threshold_rate": 0.3,
    }
    return config


    
def js_divergence(p1, p2, eps=1e-6):
    p1 = torch.clamp(p1, eps, 1.0)
    p2 = torch.clamp(p2, eps, 1.0)
    m = (p1 + p2) / 2
    kl1 = F.kl_div(m.log(), p1, reduction='none').sum()
    kl2 = F.kl_div(m.log(), p2, reduction='none').sum()
    js = (kl1 + kl2) / 2
    return js


def get_prediction_consistency(train_loader, net, device, pred_history, alpha=0.5):

    all_index = []
    all_labels = []
    all_t_label = []
    all_scores = []

    net.eval()
    with torch.no_grad():
        for imgs, txts, t_label, labels, index in train_loader:
            imgs = imgs.to(device)
            txts = txts.to(device)
            labels = labels.to(device).float()

            _, _, view1_logits, view2_logits = net(imgs, txts)
            probs = ((torch.sigmoid(view1_logits) + torch.sigmoid(view2_logits)) / 2).detach().cpu()
            labels_cpu = labels.cpu()

            for i, idx in enumerate(index):
                idx = idx.item()
                pred = probs[i]
                label = labels_cpu[i]

                loss = F.binary_cross_entropy(pred, label, reduction='mean')

                if idx not in pred_history:
                    pred_history[idx] = []
                pred_history[idx].append(loss)

                if len(pred_history[idx]) > 10:
                    pred_history[idx] = pred_history[idx][-10:]

                score = torch.tensor(pred_history[idx]).mean()
                all_scores.append(score.unsqueeze(0))

            all_index.append(index)
            all_labels.append(labels)
            all_t_label.append(t_label)

    all_index = torch.cat(all_index)
    all_labels = torch.cat(all_labels)
    all_t_label = torch.cat(all_t_label)
    all_scores = torch.cat(all_scores)

    sorted_scores, sorted_indices = torch.sort(all_scores)  
    sorted_index = all_index[sorted_indices]

    scores_std = (all_scores - all_scores.mean()) / (all_scores.std() + 1e-8)

    # update with different score
    weight = torch.tanh((1-scores_std-0.5))
    # weight = -(1-scores_std)

    weight = alpha + (1 - alpha) * weight



    return sorted_scores, sorted_index, weight.to(device)

def train(config, bit, seed,aa):
    device = config["device"]
    # alpha = config["alpha"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = (
        get_data(config)
    )
    config["num_train"] = num_train
    

    model_ft = IDCM_NN(img_input_dim=4096, text_input_dim=tag_len,output_dim=bit, num_class=n_class).to(device)

    optimizer = optim.Adam([
                            {'params': model_ft.parameters(), 'lr': 1e-4}]
                           )

    criterion = OurLoss(lambda_con=0.3,lambda_super=1.0,lambda_super_con=1.0).to(device)


    
    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    precision_list = []
    bestt2i = 0
    besti2t = 0
    n = 0

    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./PR", exist_ok=True)
    os.makedirs("./map", exist_ok=True)
    os.makedirs("./other", exist_ok=True)



    index2weight = torch.zeros(num_train).to(device)

    pred_history = defaultdict(list)
    with open(
        "./logs/data_{}_seed_{}_noiseRate_{}_bit_{}.txt".format(
            config["dataset"],
            seed,
            config["noise_rate"],
            bit,
            
        ),
        "w",
    ) as f:
        for epoch in range(config["epoch"]):
            if epoch == 30:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print(
                "%s[%2d/%2d][%s] bit:%d, dataset:%s, training...."
                % (
                    config["info"],
                    epoch + 1,
                    config["epoch"],
                    current_time,
                    bit,
                    config["dataset"],
                ),
                end="",
            )

            if (epoch + 1) % 1 == 0:
                print("calculating test binary code......")
                img_tst_binary, img_tst_label = compute_img_result(
                    test_loader, model_ft, device=device
                )
                print("calculating dataset binary code.......")
                img_trn_binary, img_trn_label = compute_img_result(
                    dataset_loader, model_ft, device=device
                )
                txt_tst_binary, txt_tst_label = compute_tag_result(
                    test_loader, model_ft, device=device
                )
                txt_trn_binary, txt_trn_label = compute_tag_result(
                    dataset_loader, model_ft, device=device
                )
                print("calculating map.......")
                t2i_mAP = calc_map_k(
                    img_trn_binary.numpy(),
                    txt_tst_binary.numpy(),
                    img_trn_label.numpy(),
                    txt_tst_label.numpy(),
                    device=device,
                )

                i2t_mAP = calc_map_k(
                    txt_trn_binary.numpy(),
                    img_tst_binary.numpy(),
                    txt_trn_label.numpy(),
                    img_tst_label.numpy(),
                    device=device,
                )


                t2i_mAP_list.append(t2i_mAP.item())
                i2t_mAP_list.append(i2t_mAP.item())
                epoch_list.append(epoch)
                print(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
                f.writelines(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f\n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )



            running_loss = 0.0

            sorted_consistency , sort_index ,weight = get_prediction_consistency(train_loader,model_ft,device,pred_history)
            model_ft.train()
            index2weight[sort_index] = weight  


            for imgs, txts, ori_labels,labels,  index in train_loader:
                optimizer.zero_grad()

                imgs = imgs.to(device)
                txts = txts.to(device)
                labels = labels.float().to(device)

                view1_feature, view2_feature,view1_predict_logit,view2_predict_logit = model_ft(imgs, txts)


                sample_weight = index2weight[index]

                loss = criterion(view1_feature, view2_feature, view1_predict_logit, view2_predict_logit, labels, sample_weight)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()



            print(
                "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f :"
                % (
                    config["info"],
                    epoch + 1,
                    bit,
                    config["dataset"],
                    config["noise_rate"],
                )
            )

            print("\b\b\b\b\b\b\b loss:%.3f" % (running_loss/len(train_loader)))

        f.writelines(
            f"best result : bit:{bit}, dataset:{config['dataset']}, noise_rate:{config['noise_rate']:.1f}, t2i_mAP:{bestt2i:.3f}, i2t_mAP:{besti2t:.3f}, average:{(besti2t + bestt2i) / 2.0 * 100.0:.3f}\n"
        )

if __name__ == "__main__":
    data_name_list = ["flickr", "nuswide21", "ms-coco","iapr"]
    bit_list = [64,128]
    noise_rate_list = [0.5,0.8]
    for data_name in data_name_list:
        for rand_num in [123]:
            for rate in noise_rate_list:
                for bit in bit_list:
                    for aa in [1]:
                        setup_seed(rand_num)

                        bit_len = bit
                        noise_rate = rate
                        dataset = data_name
                        if dataset == "nuswide21":
                            n_class = 21
                            tag_len = 1000
                        elif dataset == "flickr":
                            n_class = 24
                            tag_len = 1386
                        elif dataset == "ms-coco":
                            n_class = 80
                            tag_len = 300
                        elif dataset == "iapr":
                            n_class = 255
                            tag_len = 2912
                        config = get_config()
                        print(config)
                        train(config, bit, rand_num,aa)
                        # test(config, bit)
