"""""""""""""""""""""""""""""""""""""""""""""""""""""
Multiple Instance Class-Incremental Learning  (MICIL)
"""""""""""""""""""""""""""""""""""""""""""""""""""""
import argparse

from code_py.MICIL_test import MICIL_test
from code_py.MICIL_train import MICIL_train
from code_py.utils_MICIL import load_incremental_data

def process_train(args):

    if args.scenario == "sceE2":
        experiences = (2,0) # (training experiences, solved experiences)

        files_train = ['2_4_0_train', '5_3_1_train']
        files_valid = ['2_4_0_val', '5_3_1_val']
        files_train = ["sce_E2/" + sets for sets in files_train]
        files_valid = ["sce_E2/" + sets for sets in files_valid]

        epochs      = [15, 25]     # Number of epochs
        alphaKD_exp = [ 0, 10]     # Knowledge Distillation (KD) loss weight
        alphaL2_exp = [ 0,  1]     # Feature Matching (FM) loss weight

    if args.scenario == "sceE3":
        experiences = (3,0)

        files_train = ['2_4_train', '0_5_train', '3_1_train']
        files_valid = ['2_4_val', '0_5_val', '3_1_val']
        files_train = ["sce_E3/" + sets for sets in files_train]
        files_valid = ["sce_E3/" + sets for sets in files_valid]

        epochs      = [20, 50, 25]      # Number of epochs
        alphaKD_exp = [ 0, 10, 10]      # Knowledge Distillation (KD) loss weight
        alphaL2_exp = [ 0,  1,  1]      # Embedding Matching (FM) loss weight

    if args.scenario == "sceE5":
        experiences = (5,0)

        files_train = ['2_4_train', '0_train', '5_train', '3_train', '1_train']
        files_valid = ['2_4_val', '0_val', '5_val', '3_val', '1_val']
        files_train = ["sce_E5/" + sets for sets in files_train]
        files_valid = ["sce_E5/" + sets for sets in files_valid]

        epochs      = [20,  10,  10,  10,  10]     # Number of epochs
        alphaKD_exp = [ 0, 100, 100, 100, 100]     # Knowledge Distillation (KD) loss weight
        alphaL2_exp = [ 0,  10,  10,  10,  10]     # Embedding Matching (FM) loss weight

    # Data loading
    list_train_dataset, list_val_dataset = load_incremental_data(
        experiences = experiences[0],
        files_train = files_train,
        files_valid = files_valid,
    )

    # Model training
    MICIL_train(
        list_train_dataset = list_train_dataset,
        list_val_dataset   = list_val_dataset,
        epochs             = epochs,
        name_experiment    = args.name_experiment,
        method             = args.method,
        normalize_weights  = args.wn,
        experiences        = experiences,
        alphaKD_exp        = alphaKD_exp,
        alphaL2_exp        = alphaL2_exp,
    )

def process_test(args):

    if args.scenario == "sceE2":
        experiences = 2
        files_test = ['2_4_0_test', '5_3_1_test']
        files_test = ["sce_E2/" + sets for sets in files_test]

    if args.scenario == "sceE3":
        experiences = 3
        files_test = ['2_4_test', '0_5_test', '3_1_test']
        files_test = ["sce_E3/" + sets for sets in files_test]

    if args.scenario == "sceE5":
        experiences = 5
        files_test = ['2_4_test', '0_test', '5_test', '3_test', '1_test']
        files_test = ["sce_E5/" + sets for sets in files_test]

    MICIL_test(
        experiences     = experiences,
        files_test      = files_test,
        name_experiment = args.name_experiment,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment identifier
    parser.add_argument("--name_experiment", type=str, default="AI4SKIN_MICIL")

    # Incremental learning options
    parser.add_argument('--mode', type=str, default=2, choices=['MICIL_train', 'MICIL_test'])
    parser.add_argument('--scenario', type=str, default="sceE2", choices=['sceE2', 'sceE3', 'sceE5'])

    # Incremental learning (IL) method
    parser.add_argument('--method', type=str, default="MICIL", choices=["FT", "LwF", "MICIL"], help="Incremental learning method under MIL paradigm")
    parser.add_argument("--wn", action="store_true", help="Apply weight normalizaion")

    args = parser.parse_args()

    if args.mode == "MICIL_train":
        process_train(args=args)
    elif args.mode == "MICIL_test":
        process_test(args=args)