"""""""""""""""""""""""""""""""""""""""""""""""""""""
Multiple Instance Class-Incremental Learning  (MICIL)
"""""""""""""""""""""""""""""""""""""""""""""""""""""
import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, accuracy_score

from code_py.utils_MICIL import load_data
from code_py.utils_MIL_models import TransMIL


def process(args):

    # Data loading
    scenario = 'joint/'
    test_gen = load_data(csv_file = scenario + args.test_set)
    n_classes = len(np.unique(test_gen.dataset.targets))
    labels = np.arange(n_classes)

    # Loading model weights
    model_path = './local_data/models/model_' + args.name_experiment + '.pth'
    model = TransMIL(n_classes = n_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for it, (val_mb_x, val_mb_y) in enumerate(test_gen):

            test_logits = model(val_mb_x)[0] # Test logits
            test_logits = torch.softmax(test_logits, dim = 1) # Softmax logits
            y_pred_it = torch.argmax(test_logits, dim = 1) # Prediction

            y_pred.append(y_pred_it.item()) # Predictions
            y_true.append(val_mb_y.item()) # Ground truth

    # Test accuracy & confusion matrix
    test_acc = round(accuracy_score(y_true, y_pred),4)
    test_baacc = round(balanced_accuracy_score(y_true, y_pred),4)
    test_f1s = round(f1_score(y_true, y_pred, average='macro'),4)
    cf_mx = confusion_matrix(y_true = y_true, y_pred = y_pred, labels = labels)

    print("ACC  =", test_acc)
    print("BACC =", test_baacc)
    print("F1S  =", test_f1s)
    print(cf_mx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_experiment", type=str, default="AI4SKIN_JointTraining")
    parser.add_argument("--test_set", type=str, default="2_4_0_5_3_1_test")
    args = parser.parse_args()
    process(args=args)

if __name__ == "__main__":
    main()