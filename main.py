from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from utils.dataset_util import create_dataset
from datasets.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np
import random

def parse_option():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--device', type=str, default='cuda', help='the device you choose')
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
    parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.0001)')
    parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None)
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
    parser.add_argument('--model_type', type=str, default='ViLa_MIL', help='type of model')
    parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
    parser.add_argument('--task', type=str)
    parser.add_argument("--text_prompt", type=str, default=None)
    parser.add_argument("--text_prompt_path", type=str, default=None)
    parser.add_argument("--prototype_number", type=int, default=None)
    parser.add_argument('--csv_path', type=str, help='the csv file containing case id, slide name, and label')
    
    # select the feature type (clip-r50, conch, uni, plip, and so on)
    parser.add_argument('--clip_model_type', type=str, default='RN50', help='select the feature type')
    
    
    # ABMIL
    parser.add_argument('--pooling_strategy', type=str, default='attn', help='the pooling strategy for MIL method')
    
    
    # Text-retrevial
    parser.add_argument('--top_k_text_num', type=int, default=10, help='the num of text in the retrevial')
    
    
    # FOCUS
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--sim_threshold", type=float, default=0.8)
    
    # Selection
    parser.add_argument('--k_num', type=int, default=50)
    
    # Hyper parameter of MOE
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--num_selected', type=int, default=2)
    parser.add_argument('--topk_ratio', type=float, default=0.2)
    
    
    # Multi-Foundation Model
    parser.add_argument('--multi_fm', action='store_true', help='whether multi-foundation model')
    parser.add_argument('--multi_feature_path', nargs='+', type=str, default=[], help='path of multi-fm features')
    parser.add_argument('--num_fm', type=int, default=None, help='the num of foundation model')
    
    
    # Noise Model's parameters
    parser.add_argument('--num_iter', type=int, default=10, help='the num of iteration')
    
    return parser




def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # get the settings
    settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'mode': args.mode,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}
    
    # create the dir to store result and model weight
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.model_type), str(args.exp_code) + '_seed{}'.format(args.seed))
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

    print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir)

    settings.update({'split_dir': args.split_dir})


    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
        
        
    # load the text prompt which describes the class
    args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()
    
    # load the dataset
    dataset = create_dataset(args=args)
    
        
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    folds = np.arange(start, end)
    
    
    for i in folds:
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'test_acc': all_test_acc, 'test_f1': all_test_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                              'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
                              'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
                              'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


        
    
    


if __name__ == '__main__':
    parser = parse_option()
    args = parser.parse_args()
    
    # set the random seed 
    set_seed(args)
    
    main(args)


