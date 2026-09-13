import os
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from patch_extraction_utils import create_embeddings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_option():
    parser = argparse.ArgumentParser(description='Configurations for feature extraction')
    parser.add_argument('--patches_path', type=str, help='the path storing the patches')
    parser.add_argument('--library_path', type=str, help='the path storing the features extracted by the model')
    parser.add_argument('--model_name', type=str, help='the model which is used to extract features')
    parser.add_argument('--batch_size', type=int, help='the batch size in dataloader')
    
    return parser

if __name__ == '__main__':
    parser = parse_option()
    args = parser.parse_args()
    
    os.makedirs(args.library_path, exist_ok=True)
    create_embeddings(patch_datasets=args.patches_path, embeddings_dir=args.library_path,
                  enc_name=args.model_name, dataset='TCGA', batch_size=args.batch_size)
    
