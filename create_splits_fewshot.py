import pandas as pd 
import numpy as np 
import os 
import argparse



def parse_option():
    parser = argparse.ArgumentParser(description='split training, val, and test set for each split')
    parser.add_argument('--N', type=int, help='the training samples for each class')
    parser.add_argument('--split_num', type=int, help='the split num')
    parser.add_argument('--split_folder', type=str, help='the path of split file')
    parser.add_argument('--all_data_path', type=str, help='the path of uuid file')
    parser.add_argument('--save_folder', type=str, help='the path of save folder')
    parser.add_argument('--slide_ext', type=str, help='the extension of slide file')
    
    
    return parser


if __name__ == '__main__':
    parser = parse_option()
    args = parser.parse_args()
    
    split_num = args.split_num
    N = args.N
    data_folder = args.split_folder
    
    all_data = np.array(pd.read_excel(args.all_data_path, engine='openpyxl',  header=None, dtype={1: str}))
    save_folder = os.path.join(args.save_folder, str(N) + 'shots_' + str(split_num) + 'folds')
    
    if(not os.path.exists(save_folder)):
        os.makedirs(save_folder)
        
    for j in range(split_num):
        orginal_data_split_path = data_folder + '/splits_'+str(j)+'.csv'
        orginal_data_stastic_path = data_folder + '/splits_'+str(j)+'_descriptor.csv'
        save_path = save_folder + '/splits_'+str(j)+'.csv'

        orginal_data_split = np.array(pd.read_csv(orginal_data_split_path, dtype={1: str, 2: str, 3: str}))
        slidename2label = {}
        for each_data in all_data:   
            slidename2label[each_data[1].rstrip('.svs')] = each_data[-1]   
        all_slide_label = []
        selected_train_slide = []
        for each_data in orginal_data_split:
            slide_label = slidename2label[each_data[1]]
            all_slide_label.append(slide_label)
        all_slide_label = np.array(all_slide_label)
        unique_label = np.unique(all_slide_label)
        for each_label in unique_label:
            each_index = np.where(all_slide_label == each_label)[0]
            selected_index = np.random.choice(each_index, size=N, replace=False)
            for each_index in selected_index:
                selected_train_slide.append(orginal_data_split[each_index][1])


        orginal_data_split[:, 1][0:len(selected_train_slide)] = selected_train_slide
        orginal_data_split[:, 1][len(selected_train_slide):-1] = np.nan

        all_nums = np.array(pd.read_csv(orginal_data_stastic_path))
        val_num = np.sum(all_nums[:, 2])

        new_data_split = orginal_data_split[:val_num]

        column_name = ['','train', 'val', 'test']
        csv = pd.DataFrame(columns=column_name, data = new_data_split)
        csv.to_csv(save_path, index=False)
        
