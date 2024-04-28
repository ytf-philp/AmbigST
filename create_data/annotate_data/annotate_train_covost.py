import argparse
import pandas as pd
import numpy as np
import csv
import ast

def load_data(file_path, is_dict=False, delimiter="\t"):
    with open(file_path) as f:
        if is_dict:
            reader = csv.DictReader(f, delimiter=delimiter, quotechar=None, doublequote=False, lineterminator="\n", quoting=csv.QUOTE_NONE)
            return pd.DataFrame([dict(e) for e in reader])
        else:
            return pd.read_csv(file_path)

def prepare_phone_set(table):
    table['word'] = table['word'].apply(ast.literal_eval)
    phone_set = set()
    for words in table['word']:
        for word in words:
            phone_set.add(word)
    return phone_set

def find_phone_words(sentence, phone_set):
    found_words = [word for word in sentence[1:-1].split("*") if word in phone_set]
    return found_words if found_words else -1

def find_index(row):
    if type(row['word_text']) == str and type(row['phone']) == list:
        if row['word_text'].startswith("*"):
            row["word_text"] = row["word_text"][1:]
        return [row['word_text'].replace("'", "").lower().split("*").index(x) if x in row['word_text'].replace("'", "").lower().split("*") else -1 for x in row['phone']]
    return "-1"

def phone_idx_to_str(row):
    if isinstance(row["phone_idx"], list):
        return '|'.join(map(str, row["phone_idx"]))
    return row["phone_idx"]

def process_word_text(row):
    if isinstance(row['word_text'], float):
        return row['word_text']
    else:
        return row['word_text'].replace("'", "").split("*")

def main():
    train_df = load_data(args.train_data_path, is_dict=True)
    table = load_data(args.phone_data_path)
    phone_set = prepare_phone_set(table)
    
    train_df['phone'] = train_df['word_text'].apply(lambda x: find_phone_words(x, phone_set))
    train_df['phone_idx'] = train_df.apply(find_index, axis=1)
    train_df['idx'] = train_df.apply(phone_idx_to_str, axis=1)
    del train_df['phone_idx']
    
    train_df["word_text_process"] = train_df.apply(process_word_text, axis=1)
    del train_df["word_text_process"]
    
    train_df.to_csv(args.output_path, sep="\t", index=None, quoting=csv.QUOTE_NONE)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('train_data_path', type=str, help='Path to the training data file')
    parser.add_argument('phone_data_path', type=str, help='Path to the phone data file')
    parser.add_argument('output_path', type=str, help='Output path for the processed file')
    
    args = parser.parse_args()
    main(args)
