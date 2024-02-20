"""""""""""""""""""""""""""""""""""""""""""""""""""""
Multiple Instance Class-Incremental Learning  (MICIL)
"""""""""""""""""""""""""""""""""""""""""""""""""""""
import argparse

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from code_py.utils_MICIL import plot_training, load_data, set_random_seeds
from code_py.utils_MIL_models import TransMIL

def process(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = './local_data/models/model_' + args.name_experiment + '.pth'

    """ Data loading """
    scenario = 'joint/'
    train_dg = load_data(csv_file = scenario + '2_4_0_5_3_1_train')
    valid_dg = load_data(csv_file = scenario + '2_4_0_5_3_1_val')

    """ Variable initialization """
    best_val_acc = 0.0
    n_classes = len(np.unique(train_dg.targets)) # Number of classes
    train_acc_epoch, train_loss_epoch = [], []   # Training loss & accuracy per epoch
    val_acc_epoch, val_loss_epoch = [], []       # Validation loss & accuracy per epoch

    """ Training configuration """
    model = TransMIL(n_classes=n_classes).to(device) # TransMIL
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr) # Adam optimizer
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma) # Exponential LR decay
    class_weights = compute_class_weight('balanced', classes=np.unique(train_dg.targets), y=train_dg.targets)
    weights = torch.tensor(class_weights, dtype=torch.float).data.cuda() # Class weights
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="sum") # Weighted CE loss

    """ Training loop """
    print(f'----- Joint Training  -----')
    for ep in range(args.epochs):

        # Training loop
        run_loss_train, run_acc_train = 0.0, 0.0
        model.train()
        for it, (train_mb_x, train_mb_y) in enumerate(train_dg):
            optimizer.zero_grad()
            train_logits = model(train_mb_x)[0] # Training Forward
            loss_ce = criterion(train_logits,  train_mb_y) # WCE Loss
            loss_ce.backward() # Training Backward
            optimizer.step() # Update
            run_loss_train += loss_ce.item() # Training loss
            run_acc_train += train_mb_y.eq(torch.argmax(torch.softmax(train_logits, dim=1), dim=1)).item() # Training Accuracy
        scheduler.step()
        N = len(train_dg)
        train_loss_epoch.append(run_loss_train / N) # Keep batch size = 1
        train_acc_epoch.append(run_acc_train / N)

        # Validation loop
        model.eval()
        run_val_loss, run_val_acc = 0.0, 0.0
        with torch.no_grad():
            for it, (val_mb_x, val_mb_y) in enumerate(valid_dg):
                val_logits = model(val_mb_x)[0] # Validation Forward
                run_val_loss += criterion(val_logits, val_mb_y).item() # Validation Loss
                run_val_acc += val_mb_y.eq(torch.argmax(torch.softmax(val_logits, dim=1), dim=1)).item() # Validation Accuracy
        N = len(valid_dg)
        val_loss_epoch.append(run_val_loss / N)
        val_acc_epoch.append(run_val_acc / N)

        # Monitor loss and accuracy during training in every epoch
        print('-----------------------------------------------------------------')
        print(f'Epoch {ep+1} \t Training Loss = {train_loss_epoch[ep]:.4f} \t Validation Loss = {val_loss_epoch[ep]:.4f}')
        print(f'         \t Training Acc  = {train_acc_epoch[ep]:.4f} \t Validation Acc  = {val_acc_epoch[ep]:.4f}')
        print('-----------------------------------------------------------------')

        # Save best model
        best_acc_ep = val_acc_epoch[ep]
        if best_acc_ep > best_val_acc:
            best_val_acc = best_acc_ep
            torch.save(model.state_dict(), model_path)

    """ Plot results after each experience """
    train_metrics = [val_loss_epoch, train_loss_epoch, val_acc_epoch, train_acc_epoch]
    plot_training(metrics = train_metrics, fig_name = args.name_experiment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment identifier
    parser.add_argument("--name_experiment", type=str, default="AI4SKIN_JointTraining")

    # Training options
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='Training epochs')
    parser.add_argument('--gamma', default=0.9, type=float, help='Gamma for exponential weight decay')

    args = parser.parse_args()
    process(args=args)