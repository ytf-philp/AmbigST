import os
import csv
import argparse
import pandas as pd
import tqdm
from fairseq.data import encoders
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Process and prepare datasets")
    parser.add_argument("--root", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--lang", type=str, default="es", help="Language code")
    return parser.parse_args()

def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def load_df_from_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

def conv_calc(input_len, conv_layers):
    output_len = input_len
    for kernel, stride, padding in conv_layers:
        output_len = int((output_len + 2 * padding - (kernel - 1) - 1) / stride + 1)
    return output_len

def process_audio(word_time, n_frames):
    word_time = list(map(float, word_time.split("*")))
    conv_layers = [
        (10, 5, 0),
        (3, 2, 0),
        (3, 2, 0),
        (3, 2, 0),
        (3, 2, 0),
        (2, 2, 0),
        (2, 2, 0),
    ]
    n_hidden = conv_calc(n_frames, conv_layers)
    L, R = [], []
    L.append(0)
    total_time = word_time[-1]
    for i in range(len(word_time)):
        R.append(int(word_time[i] / total_time * n_hidden + 0.5) - 1)
        if i < len(word_time) - 1:
            L.append(R[-1] + 1)
    result = [f"{L[i]},{R[i]}" for i in range(len(word_time))]
    return '|'.join(result)

def process_text(word_text, src_text, pre_tokenizer, bpe_tokenizer):
    word_text = word_text.split("*")
    if pre_tokenizer is not None:
        src_text = pre_tokenizer.encode(src_text)
    if bpe_tokenizer is not None:
        src_text = bpe_tokenizer.encode(src_text)
    src_text = src_text.split(" ")
    L, R = [], []
    cur = 0
    for i in range(len(word_text)):
        if word_text[i] == "":
            if i == 0:
                L.append(1)
                R.append(0)
            else:
                L.append(R[i - 1] + 1)
                R.append(L[i] - 1)
        else:
            L.append(cur)
            cur = cur + 1
            while cur < len(src_text) and src_text[cur][0] != "▁":
                cur = cur + 1
            R.append(cur - 1)
    result = [f"{L[i]},{R[i]}" for i in range(len(word_text))]
    return '|'.join(result)

def main(args):
    data_dir = os.path.join(args.root, f"{args.lang}")
    config_file = Path(data_dir) / f"config_{args.lang}en.yaml"
    data_cfg = S2TDataConfig(config_file)
    dict_path = os.path.join(data_dir, data_cfg.vocab_filename)
    pre_tokenizer = encoders.build_tokenizer(Namespace(**data_cfg.pre_tokenizer))
    bpe_tokenizer = encoders.build_bpe(Namespace(**data_cfg.bpe_tokenizer))

    for split in ['train']:
        tsv_file = os.path.join(data_dir, f"{split}_raw_seg.tsv")
        df = load_df_from_tsv(tsv_file)
        data = list(df.T.to_dict().values())
        pbar = tqdm.tqdm(range(len(data)))
        for item in data:
            pbar.update()
            item["text_align"] = process_text(item["word_text"], item["src_text"], pre_tokenizer, bpe_tokenizer)
            item["audio_align"] = process_audio
