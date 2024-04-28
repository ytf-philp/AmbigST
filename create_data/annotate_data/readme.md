
# Step1: force-alignment 

(1) Download CoVoST and unzip it into ./data
(2) Learn the 
(3) run the script ./AmbigST/create_data/annotate_data/covost/process.sh
(4) Get the force-alignment dataset:
    train_raw_seg_plus.tsv
    dev_raw_seg_plus.tsv

# Step2: annotate_data 

*CoVoST dataset*
Train
```
python ./AmbigST/create_data/annotate_data/annotate_train_covost.py --input_file ./train_raw_seg_plus.tsv --output_file ./covost/fr/train_phone_homophone_seg_plus1.tsv --phone_data_path  ./covost/fr/phone.csv

```
Dev
```
python ./AmbigST/create_data/annotate_data/annotate_dev_covost.py  --input_file /zhang_m/covost/fr/dev_st.tsv  --output_file ./dev_st_homophone_seg_plus.tsv

```


*MuST-C dataset*

Train
```
python ./AmbigST/create_data/annotate_data/annotate_train_mustc.py --input_file ./train_raw_seg_plus.tsv --output_file ./covost/fr/train_phone_homophone_seg_plus1.tsv --phone_data_path  ./covost/fr/phone.csv

```
Dev
```
python ./AmbigST/create_data/annotate_data/annotate_dev_mustc.py  --input_file /zhang_m/covost/fr/dev_st.tsv  --output_file ./dev_st_homophone_seg_plus.tsv
