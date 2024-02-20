"""""""""""""""""""""""""""""""""""""""""""""""""""""
Multiple Instance Class-Incremental Learning  (MICIL)
"""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import libraries
import torch
from code_py.utils_MICIL import load_data, load_frozen_model, test_experience
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score

def MICIL_test(experiences, files_test , name_experiment):

    n_classes = 6

    seen_classes  = np.empty(0, dtype = int)
    list_classes = list(range(n_classes))

    # Loop for the incremental experiences
    list_test_dataset = []
    for task_id in range(experiences):

        """ Data loading """
        print(f'----- Loading data (exp = {task_id}) -----')
        test_set = load_data(csv_file = files_test[task_id])
        list_test_dataset.append(test_set)

        classes_exp = np.asarray(np.unique(list_test_dataset[task_id].targets), dtype=int)
        seen_classes = np.append(seen_classes, classes_exp)
        unseen_classes = list(set(list_classes) - set(seen_classes))

        solved_model_exp = 'model_exp' + str(task_id) + '_' + name_experiment
        model = load_frozen_model(name = solved_model_exp, n_classes = n_classes)

        print(f'----- Testing (exp = {task_id}) -----') # Testear todas las experiencias del modelo
        true_y_exp, preds_y_exp = np.empty(0, dtype=int), np.empty(0, dtype=int)
        for exp, test_dataset in enumerate(list_test_dataset[0:task_id+1]):
            _,true_y_exp_e, preds_y_exp_e =test_experience(
                model = model,
                test_dataset = test_dataset,
                unseen_classes = unseen_classes,
            )
            true_y_exp = np.append(true_y_exp, true_y_exp_e)
            preds_y_exp = np.append(preds_y_exp, preds_y_exp_e)

        preds_y_exp = np.squeeze(np.stack(preds_y_exp))
        true_y_exp = np.squeeze(np.stack(true_y_exp))
        test_acc = accuracy_score(y_true=true_y_exp, y_pred=preds_y_exp)
        test_baacc = balanced_accuracy_score(y_true=true_y_exp, y_pred=preds_y_exp)
        test_f1s_score = f1_score(y_pred=preds_y_exp, y_true=true_y_exp, labels=np.unique(true_y_exp).tolist(), average='macro')
        cm = confusion_matrix(y_true=true_y_exp, y_pred=preds_y_exp, labels=[0, 1, 2, 3, 4, 5])

        print("--- VALIDATION IN ALL EXPERIENCES ---")
        print("ACC  =", round(test_acc,4))
        print("BACC =", round(test_baacc,4))
        print("F1S  =", round(test_f1s_score,4))
        print(cm)