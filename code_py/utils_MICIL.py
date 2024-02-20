"""""""""""""""""""""""""""""""""""""""""""""""""""""
Multiple Instance Class-Incremental Learning  (MICIL)
"""""""""""""""""""""""""""""""""""""""""""""""""""""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

from code_py.utils_MIL_data import MILDataset, MILDataGenerator
from code_py.utils_MIL_models import TransMIL


# Code to load bags for MIL
def load_data(csv_file):
    """
    LOAD_DATA: Load bags for Multiple-Instance Learning (MIL). Batch size = 1, by default.
        + Inputs:
            - csv_file: csv file with the WSI data for a unique experience
            - n_ims: number of patches per WSI because of memory limitations
        + Outputs:
            - datagenerator: DataGenerator object for iteration in epochs loops
    """

    file_name = './local_data/csv/' + csv_file + '.csv'
    dataframe = pd.read_csv(filepath_or_buffer = file_name, dtype = str,  delimiter = ',')
    dataset = MILDataset(data_frame = dataframe)
    datagenerator = MILDataGenerator(dataset = dataset)
    return datagenerator

def load_incremental_data(experiences, files_train, files_valid):
    list_train_dataset, list_val_dataset = [], []
    for task_id in range(experiences):
        print(f'----- Loading data (exp = {task_id}) -----')
        train_gen = load_data(csv_file = files_train[task_id])
        valid_gen = load_data(csv_file = files_valid[task_id])
        list_train_dataset.append(train_gen)
        list_val_dataset.append(valid_gen)
    return list_train_dataset, list_val_dataset

def get_old_outputs(train_dataset, old_model, old_classes):
    N = train_dataset.__len__()  # Number of images in training dataset
    L = old_model.final_fc[0].in_features  # Length of the latent space
    n_old = len(old_classes)  # Number of classes

    old_logits = torch.empty(N, n_old)  # (N x nº old classes)
    old_features = torch.empty(N, L)  # (N x feature length)
    for it, (train_mb_x, train_mb_y) in enumerate(train_dataset):
        prev_logits, prev_features = old_model(train_mb_x)
        old_logits[it] = prev_logits[:, old_classes]
        old_features[it] = prev_features
    return old_logits, old_features


def kd_loss(prev_logits, train_logits, ta = 2):
    """
    KD_LOSS: Compute distillation loss between output of the current model and the output of the previous (saved) model.
        + Inputs:
            - prev_logits: Logits of model previous experience only for active units
            - train_logits: Logits of model in training phase only for active units
        + Outputs:
            - dist_loss: Knowledge distrillation loss
    """
    q = torch.softmax(prev_logits / ta, dim = 0).cuda().unsqueeze(dim=0)
    log_p = torch.log_softmax(train_logits / ta, dim = 1)
    dist_loss = torch.nn.functional.kl_div(log_p, q, reduction = "batchmean")
    return dist_loss

def test_experience(model, test_dataset, unseen_classes):
    run_test_acc = 0.0
    preds_y, true_y, list_prob = [], [], []

    model.eval()
    with torch.no_grad():
        for it, (val_mb_x, val_mb_y) in enumerate(test_dataset):

            # Probabilities and prediction
            test_logits = model(val_mb_x)[0]
            test_logits[:,unseen_classes] = -100
            test_logits = torch.softmax(test_logits, dim = 1)
            y_pred = torch.argmax(test_logits, dim = 1)
            run_test_acc += val_mb_y.eq(y_pred).item()

            # Save list of probabilities, prediction and ground truth
            list_prob.append(np.max(test_logits.data.to('cpu').numpy()))
            preds_y.append(y_pred.item())
            true_y.append(val_mb_y.item())

    # Test accuracy & confusion matrix
    test_acc_exp = round(run_test_acc/len(test_dataset),4)
    test_baacc = round(balanced_accuracy_score(true_y, preds_y), 4)
    test_f1s_score = round(f1_score(y_pred=preds_y, y_true=true_y, labels = np.unique(true_y).tolist(), average='macro'),4)
    cm = confusion_matrix(y_true = true_y, y_pred = preds_y, labels = [0, 1, 2, 3, 4, 5])

    print("ACC  =", test_acc_exp)
    print("BACC =", test_baacc)
    print("F1S  =", test_f1s_score)
    print(cm)
    return test_acc_exp, true_y, preds_y

# Load pretrained model weights
def load_model_weights(model_name, n_classes):
    model = TransMIL(n_classes = n_classes)
    state_dict = torch.load('./local_data/models/' + model_name + '.pth')
    model.load_state_dict(state_dict)
    return model.cuda()

# Laoa frozen teacher model
def load_frozen_model(name, n_classes):
    previous_model = load_model_weights(model_name = name, n_classes = n_classes)
    for param in previous_model.parameters():
        param.requires_grad = False
    return previous_model

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

def plot_val(metrics, fig_name):

    # Accuracy per epoch and experience
    acc = []
    epochs = len(metrics[1])
    sets = len(metrics[1][0])
    for i in metrics[1]:
        for j in i:
            acc.append(j)
    acc = np.array(acc).reshape([epochs, sets]).T

    # Plot validation accuracy
    plt.figure(figsize = (10, 5))
    plt.plot(metrics[0], label = "global")
    for exp in range(sets):
        plt.plot(acc[exp], label = "exp. " + str(exp))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig('./local_data/figures/' + fig_name + '.png') # Save figure

def plot_training(metrics, fig_name):
    # Create subplots
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # Plot losses
    ax1.plot(metrics[0], label = "val")
    ax1.plot(metrics[1], label = "train")

    # Distillation losses
    if len(metrics) > 4:
        ax1.plot(metrics[4], label = "CE")
    if len(metrics) > 5:
        ax1.plot(metrics[5], label = "KD")
    if len(metrics) > 6:
        ax1.plot(metrics[6], label = "L2")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot accuracy
    ax2.plot(metrics[2], label = "val")
    ax2.plot(metrics[3], label = "train")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.savefig('./local_data/figures/' + fig_name + '.png') # Save figure