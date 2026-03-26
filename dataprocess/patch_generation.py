# this code utilzes the coords of patch to split each slide into some patches
# you should input some nessary information including the path of original slides
# the coords of each patches, the patch size of each patch, and save folder path. 
import os
import h5py
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse


def generate_patch(
    patch_file_name,
    slide_folder,
    scale,
    slide_ext, 
    is_uuid_inpath,
):
    """
    this function splits the slide into some patch with h5 file which contains the coords information.
    Args:
        patch_file_name: the file name for each h5 file containing the coord information.
        slide_folder: the folder which contains slides.
        scale: the scale number for each patch.
        slide_ext: the extension for the slide file.
        is_uuid_inpath: whether the uuid is in path.
    """
    patch_path = os.path.join(patch_folder, patch_file_name)
    if is_uuid_inpath:
        slide_path = os.path.join(slide_folder, svs2uuid[patch_file_name.replace('h5', 'svs')], patch_file_name.replace('h5', 'svs'))
    else:
        slide_path = os.path.join(slide_folder, patch_file_name.replace('h5', slide_ext))

    f = h5py.File(patch_path, 'r')
    coords = f['coords']
    patch_level = coords.attrs['patch_level']
    patch_size = coords.attrs['patch_size']
    slide = openslide.open_slide(slide_path)
    try:
        magnification = int(float(slide.properties['aperio.AppMag']))
    except:
        magnification = 40
    save_path = save_folder + patch_file_name.replace('.h5', '')
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    if(magnification == 20):
        resized_patch_size = int(patch_size/(scale/2))
    else:
        resized_patch_size = int(patch_size/scale)
    for coord in tqdm(coords, leave=False):
        coord = (int(coord[0]), int(coord[1]))
        patch = slide.read_region(coord, int(patch_level), (int(patch_size), int(patch_size)))
        patch = patch.resize((resized_patch_size, resized_patch_size))
        patch_name = str(coord[0]) + '_' + str(coord[1]) + '.png'
        patch_save_path = os.path.join(save_path, patch_name)
        patch.save(patch_save_path)


def parse_option():
    parser = argparse.ArgumentParser(description='patch generation')
    parser.add_argument('--slide_folder', type = str, help='the path of origenal slides')
    parser.add_argument('--uuid_file_path', type=str, help='the path of uuid file')
    parser.add_argument('--root_path', type=str, help='the path of folder which stores the coord information')
    parser.add_argument('--save_path', type=str, help='the path of save file')
    parser.add_argument('--patch_size', type=int, help='the size of each patch')
    parser.add_argument('--slide_ext', type=str, help='the extension of slide file')
    
    return parser

if __name__ == '__main__':
    parser = parse_option()
    args = parser.parse_args()
    slide_folder = args.slide_folder
    all_data = np.array(pd.read_excel(args.uuid_file_path, engine='openpyxl',  header=None))
    root_folder = args.root_path
    patch_folder = root_folder + '/patches_' + str(args.patch_size) + '/'
    save_folder = args.save_path
    slide_ext = args.slide_ext
    is_uuid_inpath = False
    define_patch_size = args.patch_size
    
    
    if(define_patch_size == 2048):
        scale = 8
        save_name = '5x'
    elif(define_patch_size == 1024):
        scale = 4
        save_name = '10x'
    elif(define_patch_size == 512):
        scale = 2
        save_name = '20x'

    if(not os.path.exists(root_folder)):
        os.makedirs(root_folder)

    svs2uuid = {}
    for i in all_data:
        svs2uuid[i[1].rstrip('\n')] = i[0]
    
    pool = ThreadPoolExecutor(max_workers=16)
    all_file_names = os.listdir(patch_folder)
    for patch_file_name in all_file_names:
        pool.submit(generate_patch, patch_file_name, slide_folder, scale, slide_ext, is_uuid_inpath)




