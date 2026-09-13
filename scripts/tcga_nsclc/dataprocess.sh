# written by Jiahao Xu on 2025.7.13
# generate the split and N-shot file

# generate the uuid csv
python dataprocess/ProcessC16.py \
--dataset tcga_nsclc \
--original_csv_file /data3/Public/TCGA_NSCLC/label.csv \
--save_csv_path /data3/Public/TCGA_NSCLC/slide_name_file.csv \
--save_uuid_csv_path /data3/Public/TCGA_NSCLC/slide_name_file_uuid.xlsx \


# create the splits of camelyon all dataset
python create_splits_seq.py \
--label_frac 1 \
--k 10 \
--task 'task_tcga_lung_subtyping' \
--val_frac 0.3 \
--test_frac 0.3 \
--dataset tcga_nsclc \
--csv_path /data3/Public/TCGA_NSCLC/slide_name_file.csv \

# create the N-shot for each split
python create_splits_fewshot.py \
--N 1 \
--split_num 10 \
--split_folder splits/tcga_nsclc/seed=1/splits10/datasplit \
--all_data_path /data3/Public/TCGA_NSCLC/slide_name_file_uuid.xlsx \
--save_folder splits/tcga_nsclc/seed=1/splits10 \
--slide_ext .tif \