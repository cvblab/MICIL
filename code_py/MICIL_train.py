"""""""""""""""""""""""""""""""""""""""""""""""""""""
Multiple Instance Class-Incremental Learning  (MICIL)
"""""""""""""""""""""""""""""""""""""""""""""""""""""
import random
import warnings

import numpy as np
import torch
from sklearn.utils import compute_class_weight

from code_py.utils_MICIL import load_frozen_model, load_model_weights, plot_training, plot_val, test_experience, kd_loss
from code_py.utils_MIL_models import TransMIL

def MICIL_train(name_experiment, list_train_dataset, list_val_dataset, normalize_weights, method, experiences, alphaKD_exp, alphaL2_exp, epochs):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_classes = 6 # Total number of classes
    list_lr = [1e-5, 1e-6, 1e-6, 1e-6, 1e-6]  # Learning rate per experience

    # Arrays to handle previoys and actual classes
    list_classes = list(range(n_classes))
    old_classes  = np.empty(0, dtype = int)
    seen_classes = np.empty(0, dtype = int)

    """ Loop for the incremental experiences """
    solved_models = []
    for task_id in range(experiences[0]):
        experiment_name = 'exp' + str(task_id) + '_' + str(name_experiment) # Task ID + experiment ID
        model_path = './local_data/models/model_' + experiment_name + '.pth'

        # Variable initialization
        train_acc_epoch, train_loss_epoch = [], []  # Training loss & accuracy per epoch
        val_acc_epoch, val_loss_epoch = [], []      # Validation accuracies per epoch in current experience
        global_acc_val, val_acc_exp_epoch = [], []  # Validation loss & accuracy per epoch and experience
        ce_loss_epoch, kd_loss_epoch, l2_loss_epoch = [], [], []  # Distillation losses

        exp_classes = np.asarray(np.unique(list_val_dataset[task_id].targets), dtype=int)  # Classes in current experience
        seen_classes = np.append(seen_classes, exp_classes)  # Classes in current experience
        unseen_classes = list(set(list_classes) - set(seen_classes)) # Inactive units

        if task_id < experiences[1]:
            solved_models.append('model_' + experiment_name)
            old_classes = np.append(old_classes, exp_classes)
            continue

        # Model building
        print(f'----- Loading model (exp = {task_id}) -----')
        if task_id > 0:
            old_model = load_frozen_model(n_classes = n_classes, name = solved_models[task_id-1]) # Model from previous experience
            model = load_model_weights(model_name = solved_models[task_id-1], n_classes = n_classes) # Model for training
        else:
            model = TransMIL(n_classes = n_classes).to(device)  # MIL Aggregation w/ Transformers

        # Evaluation datasets of previously and actual experiences
        if task_id > 0:
            print(f'----- Initial validation (exp = {task_id}) -----')
            for exp, test_dataset in enumerate(list_val_dataset[0:task_id+1]):
                ini_acc_val,_,_ = test_experience(model = model, test_dataset = test_dataset, unseen_classes = unseen_classes)

        # Hyperparameter selection
        train_dataset = list_train_dataset[task_id]
        alpha_kd, alpha_l2 = alphaKD_exp[task_id], alphaL2_exp[task_id]

        optimizer = torch.optim.Adam(model.parameters(), lr = list_lr[task_id]) # Adam optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = 0.9)

        # Weighted CE loss
        class_weights = np.zeros(n_classes, dtype=float)
        targets = train_dataset.targets
        exp_targets = np.array([element for element in targets if int(element) in exp_classes])
        class_weights[exp_classes] = compute_class_weight('balanced', classes=np.unique(exp_targets), y=exp_targets)
        mem_targets = np.array([element for element in targets if int(element) in old_classes])
        if len(mem_targets)>0:
            class_weights[old_classes] = compute_class_weight('balanced', classes=np.unique(mem_targets), y=mem_targets)
        weights = torch.tensor(class_weights, dtype=torch.float).data.cuda()  # Class weights
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="sum")  # Weighted CE loss

        # Outputs of the teacher model
        if task_id > 0:
            N = train_dataset.__len__()  # Number of images in training dataset
            L = old_model.final_fc[0].in_features # Length of the latent space
            n_old = len(old_classes) # Number of classes

            old_logits = torch.empty(N, n_old) # (N x nº old classes)
            old_features = torch.empty(N, L) # (N x feature length)
            for it, (train_mb_x, train_mb_y) in enumerate(train_dataset):
                prev_logits, prev_features = old_model(train_mb_x)
                old_logits[it] = prev_logits[:, old_classes]
                old_features[it] = prev_features

        # Last FC weight normalization
        if normalize_weights & (task_id>0):
            with torch.no_grad():
                model.final_fc[0].weight.data = torch.nn.functional.normalize(model.final_fc[0].weight.data)

        """ Training loop """
        print(f'----- Training experience (exp = {task_id}) -----')
        for ep in range(epochs[task_id]):

            run_loss_train, run_acc_train = 0.0, 0.0
            run_loss_kd, run_loss_l2, run_loss_ce = 0.0, 0.0, 0.0
            model.train()
            for it, (train_mb_x, train_mb_y) in enumerate(train_dataset):
                optimizer.zero_grad()
                train_logits, train_features = model(train_mb_x)
                train_logits[:, unseen_classes] = -100 # Mask inactive units
                loss_ce = criterion(train_logits,  train_mb_y)
                run_loss_ce += loss_ce.item()

                if task_id > 0:

                    # Knowledge Distillation (KD) loss
                    if alpha_kd != 0:
                        loss_kd = kd_loss(prev_logits=old_logits[it], train_logits=train_logits[:, old_classes])
                        run_loss_kd += alpha_kd * loss_kd.item()

                    # Embedding Matching (EM)loss
                    if alpha_l2 != 0:
                        criterion_l2 = torch.nn.MSELoss()
                        loss_l2 = criterion_l2(train_features.squeeze(dim=0), old_features[it].cuda())
                        run_loss_l2 += alpha_l2 * loss_l2.item()

                    # Incremental learning loss
                    if method == "FT":
                        loss = loss_ce  # CE loss (fine-tunning)
                    elif method == "LwF":
                        loss = loss_ce + alpha_kd * loss_kd  # CE + KD loss (LwF)
                    elif method == "MICIL":
                        loss = loss_ce + alpha_kd * loss_kd + alpha_l2 * loss_l2  # CE + KD + L2 loss (MICIL)
                else:
                    loss = loss_ce

                loss.backward()
                optimizer.step()
                run_loss_train += loss.item()
                run_acc_train += train_mb_y.eq(torch.argmax(torch.softmax(train_logits, dim=1), dim=1)).item()

                # Weight normalization
                if normalize_weights & (task_id>0):
                    with torch.no_grad():
                            model.final_fc[0].weight.data = torch.nn.functional.normalize(model.final_fc[0].weight.data)

            scheduler.step()
            N = len(train_dataset) # Keep batch size = 1
            train_loss_epoch.append(run_loss_train / N)
            train_acc_epoch.append(run_acc_train / N)
            ce_loss_epoch.append(run_loss_ce / N)
            kd_loss_epoch.append(run_loss_kd / N)
            l2_loss_epoch.append(run_loss_l2 / N)

            # Validation loop in each experience
            model.eval()
            val_loss_exp, val_acc_exp = [], []
            global_run_val_acc, global_n = 0.0, 0.0

            for exp, val_dataset in enumerate(list_val_dataset[0:task_id+1]): # Validate ALSO in previous experiences
                run_val_loss, run_val_acc = 0.0, 0.0
                with torch.no_grad():
                    for it, (val_mb_x, val_mb_y) in enumerate(val_dataset):
                        val_logits = model(val_mb_x)[0]
                        val_logits[:, unseen_classes] = -100 # Mask inactive units
                        run_val_loss += criterion(val_logits, val_mb_y).item() # Validation Loss & Accuracy
                        run_val_acc += val_mb_y.eq(torch.argmax(torch.softmax(val_logits, dim=1), dim=1)).item()

                val_loss_exp.append(run_val_loss/len(val_dataset))
                val_acc_exp.append(run_val_acc/len(val_dataset)) # Validation accuracy per experience
                global_run_val_acc += run_val_acc
                global_n += len(val_dataset) # Dataset size of individual experience

            # Save validation metrics of current experience
            val_loss_epoch.append(val_loss_exp[task_id])
            val_acc_epoch.append(val_acc_exp[task_id])
            val_acc_exp_epoch.append(val_acc_exp)
            global_acc_val.append(global_run_val_acc/global_n)

            # Monitor loss and accuracy during training in every epoch
            print('-------------------------------------------------------------------------')
            print(f'Epoch {ep+1} \t Training Loss = {train_loss_epoch[ep]:.4f} \t Validation Loss = {val_loss_epoch[ep]:.4f}')
            print(f'         \t Training Acc  = {train_acc_epoch[ep]:.4f} \t Validation Acc  = {val_acc_epoch[ep]:.4f}')
            print('-------------------------------------------------------------------------')

            # Print validation accuracy in multiple experiences
            if task_id > 0:
                for exp_n in range(task_id+1):
                    print(f'Experience {exp_n}. Validation accuracy = {val_acc_exp[exp_n]:.4f}')
                print(f'Global set  . Validation accuracy = {global_acc_val[ep]:.4f}')

        torch.save(model.state_dict(), model_path) # Saving model (early stopping on first experience)
        solved_models.append('model_' + experiment_name)
        old_classes = np.append(old_classes, exp_classes)

        """ Plot results after each experience """
        if task_id == 0:
            train_metrics = [val_loss_epoch, train_loss_epoch, val_acc_epoch, train_acc_epoch]
            plot_training(metrics=train_metrics, fig_name=experiment_name)
        else:
            plot_val(metrics = [global_acc_val, val_acc_exp_epoch], fig_name = experiment_name)
        torch.cuda.empty_cache()