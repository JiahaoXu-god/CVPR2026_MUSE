import os
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits
import argparse
import numpy as np



def parse_option():
    parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
    
    parser.add_argument('--label_frac', type=float, default= 1.0, help='fraction of labels (default: 1)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of splits (default: 10)')
    parser.add_argument('--task', type=str)
    parser.add_argument('--val_frac', type=float, default= 0.2, help='fraction of labels for validation (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default= 0.2, help='fraction of labels for test (default: 0.1)')
    parser.add_argument('--csv_path', type=str, help='csv file containing case id, slide id, and label')
    parser.add_argument('--dataset', type=str, help='the dataset you choose')

    return parser

if __name__ == '__main__':
    parser = parse_option()
    args = parser.parse_args()
    
    if args.task == 'task_tcga_rcc_subtyping':
        label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2}
        n_classes = 3
        
    elif args.task == 'task_tcga_lung_subtyping':
        label_dict = {'LUAD':0, 'LUSC':1}
        n_classes = 2
    elif args.task == 'task_tcga_brca_subtyping':
        label_dict = {'IDC': 0, 'ILC':1}
        n_classes = 2
    elif args.task in {'task_camelyon16_binary', 'task_camelyon_all_binary'}:
        label_dict = {'normal':0, 'tumor':1}
        n_classes = 2
    elif args.task == 'task_ubc_ocean_subtyping':
        label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        n_classes = 5
    
    else:
        raise NotImplementedError
    
    
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = label_dict,
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.round(num_slides_cls * args.val_frac).astype(int)
    test_num = np.round(num_slides_cls * args.test_frac).astype(int)
    
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/' + args.dataset + '/' + 'seed={}'.format(int(args.seed)) + '/' + 'splits{}'.format(int(args.k)) + '/' + 'datasplit'
        #split_dir = 'splits/'+ args.dataset +'/'+ str(args.task) + '_seed{}'.format(int(args.seed)) + '_splits{}'.format(int(args.k))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
    




