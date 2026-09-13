from datasets.dataset_generic import Generic_MIL_Dataset
from datasets.dataset_generic_multi import Multi_Generic_MIL_Dataset
import os

def create_dataset(args):
    """
    create the dataset according to the arguments.
    Args:
        args: the arguments.
    Return:
        dataset: the dataset.
    """
    
    if args.task == 'task_tcga_rcc_subtyping':
        label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2}
        args.n_classes = 3
    
    elif args.task == 'task_tcga_lung_subtyping':
        label_dict = {'LUAD':0, 'LUSC':1}
        args.n_classes = 2
        
    elif args.task == 'task_tcga_brca_subtyping':
        label_dict = {'IDC': 0, 'ILC':1}
        args.n_classes = 2
    
    elif args.task in {'task_camelyon16_binary', 'task_camelyon_all_binary'}:
        label_dict = {'normal':0, 'tumor':1}
        args.n_classes = 2
    elif args.task == 'task_ubc_ocean_subtyping':
        label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        args.n_classes = 5
        
    else:
        raise NotImplementedError
    
    if args.multi_fm:
        data_dir_list = args.multi_feature_path
        args.num_fm = len(data_dir_list)
        dataset = Multi_Generic_MIL_Dataset(csv_path = args.csv_path,
                                    mode = args.mode,
                                    data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                    data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                    data_dir_list = data_dir_list, 
                                    shuffle = False,
                                    print_info = True,
                                    label_dict = label_dict,
                                    patient_strat= False,
                                    ignore=[])
    else:
        
        dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                                    mode = args.mode,
                                    data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                    data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                    shuffle = False,
                                    print_info = True,
                                    label_dict = label_dict,
                                    patient_strat= False,
                                    ignore=[])
    
    return dataset