# written by Jiahao Xu on 2025.7.13
# generate the split and N-shot file

# create the splits of camelyon all dataset
python create_splits_seq.py \
--label_frac 1 \
--k 10 \
--task 'task_camelyon_all_binary' \
--val_frac 0.3 \
--test_frac 0.3 \
--dataset camelyon_all \
--csv_path /data3/Public/CAMELYON_ALL/slide_name_file.csv \

# create the N-shot for each split
python create_splits_fewshot.py \
--N 5 \
--split_num 10 \
--split_folder splits/camelyon_all/seed=1/splits10/datasplit \
--all_data_path /data3/Public/CAMELYON_ALL/slide_name_file_uuid.xlsx \
--save_folder splits/camelyon_all/seed=1/splits10 \
--slide_ext .tif \

