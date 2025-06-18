# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import sys
sys.path.append(".")  # Ensure current dir is in path

from alg.opt import *
from alg import alg, modelopera
from utils.util import (
    set_random_seed,
    get_args,
    print_row,
    print_args,
    train_valid_target_eval_names,
    alg_loss_dict,
    print_environ,
    save_checkpoint  # Added import
)
from datautil.getdataloader_single import get_act_dataloader
from uci_loader import get_uci_har_dataloader

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)

    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    if args.dataset == 'uci_har':
        args.num_classes = 6

    if args.dataset == 'uci_har':
        train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_uci_har_dataloader(args)
    else:
        train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)
        if hasattr(args, 'domain_num'):
            args.latent_domain_num = args.domain_num

    print(f"[CONFIG] Using {args.latent_domain_num} domains with {args.num_classes} classes")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    best_valid_acc, best_target_acc = 0, 0
    best_model_state = None

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm._initialize_dclassifier(train_loader)
    algorithm.train()

    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n======== ROUND {round} ========')

        print('==== Feature update ====')
        loss_list = ['class']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15)

        print('==== Latent domain characterization ====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('==== Domain-invariant feature learning ====')
        loss_list = alg_loss_dict(args)
        print_key = ['epoch'] + [item + '_loss' for item in loss_list] + ['train_acc', 'total_cost_time']
        print_row(print_key, colwidth=15)

        sss = time.time()
        algorithm.train()  # Ensure training mode

        for step in range(args.local_epoch):
            for data in train_loader:
                step_vals = algorithm.update(data, opt)
            
            train_acc = modelopera.accuracy(algorithm, train_loader_noshuffle, None)
            
            results = {
                'epoch': step,
                'train_acc': train_acc,
                'total_cost_time': time.time() - sss,
            }
            for key in loss_list:
                results[key + '_loss'] = step_vals[key]
            
            print_row([results[key] for key in print_key], colwidth=15)
        
        # === VALIDATION PHASE ===
        algorithm.eval()  # Switch to evaluation mode
        with torch.no_grad():
            valid_acc = modelopera.accuracy(algorithm, valid_loader, None)
            target_acc = modelopera.accuracy(algorithm, target_loader, None)
        
        print(f"\n[VALIDATION] Round {round}:")
        print(f"  Valid Accuracy: {valid_acc:.4f}")
        print(f"  Target Accuracy: {target_acc:.4f}")
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_target_acc = target_acc
            best_model_state = algorithm.state_dict()
            checkpoint_path = os.path.join(args.output_dir, f'best_model_round{round}.pth')
            save_checkpoint({
                'round': round,
                'state_dict': best_model_state,
                'best_valid_acc': best_valid_acc,
                'best_target_acc': best_target_acc,
            }, filename=checkpoint_path)
            print(f"🔥 NEW BEST MODEL: Saved to {checkpoint_path}")
            print(f"  Valid Acc: {best_valid_acc:.4f}, Target Acc: {best_target_acc:.4f}")

    # === FINAL TESTING ===
    print("\n==== FINAL TESTING ====")
    if best_model_state:
        algorithm.load_state_dict(best_model_state)
        algorithm.eval()
        with torch.no_grad():
            final_valid_acc = modelopera.accuracy(algorithm, valid_loader, None)
            final_target_acc = modelopera.accuracy(algorithm, target_loader, None)
        
        print(f"Best Model Performance:")
        print(f"  Valid Accuracy: {final_valid_acc:.4f}")
        print(f"  Target Accuracy: {final_target_acc:.4f}")
        
        # Save final metrics
        with open(os.path.join(args.output_dir, 'final_metrics.txt'), 'w') as f:
            f.write(f"Best Valid Accuracy: {final_valid_acc:.4f}\n")
            f.write(f"Best Target Accuracy: {final_target_acc:.4f}\n")
    else:
        print("No best model found for final testing")

    print(f"\nTraining complete! Best target accuracy: {best_target_acc:.4f}")

if __name__ == '__main__':
    args = get_args()
    main(args)
