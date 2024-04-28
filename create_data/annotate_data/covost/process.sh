

covost_data=./data/fr
data_workspace=./data/fr/covost_fr
data_jsonl=./data/fr/dataset_train.jsonl
data_segment=./data/fr/segment/train
data_align=./data/fr/segment/train_align
grid_json=./data/fr/dataset_train_raw.jsonl



echo "Extract Data"

python /data/ytf/AmbigST/create_data/annotate_data/covost/extract_covost.py --data_dir $covost_data --output_dir $data_workspace

echo "Clean Data"

python /data/ytf/AmbigST/create_data/annotate_data/covost/clean_data.py --dataset_path $data_workspace --output_jsonl $data_jsonl --audio_output_folder $data_segment

echo "Forced Alignment"
#download dictionary and acoustic model from MFA
mfa align --clean $data_segment french_mfa french_mfa $data_align

echo "Combine Textgrid"

python  /data/ytf/AmbigST/create_data/annotate_data/covost/write_textgrid.py --input_file $data_jsonl --output_file $grid_json --text_grid_folder $data_align

echo "Word_align_info_raw"

python /data/ytf/AmbigST/create_data/annotate_data/covost/word_align_info_raw.py  --root ./data --lang fr

echo "Combine"

echo "Finish preprocess!"

