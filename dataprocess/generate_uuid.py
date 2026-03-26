import uuid
import pandas as pd
import argparse

def process_label_file(
    dataset, 
    label_path, 
    save_csv_path, 
    save_uuid_excel_path
):
    """
    this function generates the csv file according to the csv file which 
    contains file names and labels.
    Args:
        label_path: the csv file containing the slide names and labels.
        save_csv_path: the csv file containing case id, slide id, and label.
        save_uuid_excel_path: the csv file containing uuid, slide id, and labels.
    """
    # 读取原始标签文件，无列名
    if dataset == 'ubc_ocean':
        df = pd.read_csv(label_path, header=None, names=['slide_id', 'label', 'state'], dtype={0: str})
    else:
        df = pd.read_csv(label_path, header=None, names=['slide_id', 'label'])

    # 映射标签为字符串
    if dataset == 'camelyon_all':
        label_mapping = {0: 'normal', 1: 'tumor'}
        df['label'] = df['label'].map(label_mapping)

    # 生成 case_id
    df['case_id'] = ['patient_{}'.format(i) for i in range(len(df))]

    # 保存为 ViLa-MIL 所需的 CSV 格式：case_id, slide_id, label
    df_csv = df[['case_id', 'slide_id', 'label']]
    
    df_csv.to_csv(save_csv_path, index=False)

    # 生成 uuid
    df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]

    # 保存为 Excel 文件：uuid, slide_id, label_str
    df_uuid = df[['uuid', 'slide_id', 'label']]
    df_uuid.to_excel(save_uuid_excel_path, index=False, header=False)

    
    


def parse_option():
    parser = argparse.ArgumentParser(description='generate the csv file which contains the uuid')
    parser.add_argument('--dataset', type=str, help='the dataset you want to process')
    parser.add_argument('--original_csv_file', type=str, help='the original csv file')
    parser.add_argument('--save_csv_path', type=str, help='case id, slide id and label')
    parser.add_argument('--save_uuid_csv_path', type=str, help='uuid, slide id and label')
    
    return parser


if __name__ == "__main__":
    parser = parse_option()
    args = parser.parse_args()
    
    setting = {
        'dataset': args.dataset,
        'label_path': args.original_csv_file, 
        'save_csv_path': args.save_csv_path, 
        'save_uuid_excel_path': args.save_uuid_csv_path
    }
    
    process_label_file(**setting)
    
    
