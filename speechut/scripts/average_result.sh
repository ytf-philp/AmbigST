cd /home/SpeechUT
model_dir=/zhang_m/model/Baseline_model
python fairseq/scripts/average_checkpoints.py --inputs $model_dir/checkpoint.best_acc*.pt --output $model_dir/checkpoint.avgnbest.pt