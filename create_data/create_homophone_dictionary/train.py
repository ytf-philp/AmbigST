import pandas as pd
import numpy as np
import os
import yaml
import string
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


#读取文件，并进行初步处理
def match_strict(sentence,word):

    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    word = word.translate(str.maketrans('', '', string.punctuation))
    sentence=sentence.split(" ")
    if word in sentence:
        return True
    else:  
        return False

#删除数字
def remove_digits_from_string(s):
    strr=""
    for x in s:
        if not x.isdigit():
            strr=strr+x
    return strr

#删除数字
def remove_special_token_from_string(s):
    strr=""
    for x in str(s):
        if x!="'":
            strr=strr+x
    return strr

#删除 word单词长度小于2的列
def remove_column(table):
    for index in range(len(table["word"])):
        if len(table["word"][index])<2 or table["word"][index].startswith("'") or table["word"][index].startswith("\""):
            table.drop(index=index,inplace=True)     

#删除大于3的列
def remove_multi_word(table):
    for index in range(len(table["word"])):
        if len(table["word"][index])>2:
            table.drop(index=index,inplace=True)



def main(args):
    table = pd.read_csv(args.input_file, sep="\t")
    # 删除以标点符号开头的
    table["phone"] = table["phone"].apply(remove_digits_from_string)
    table["word"] = table["word"].apply(remove_special_token_from_string)
    data_dict = table.groupby('phone').apply(lambda x: {col: list(set(x[col])) for col in x.columns if col != 'phone'}).to_dict()
    # 对文件进行合并
    # 对合并的文件进行筛选处理
    result = {"phone": [], "word": []}
    for key, value in data_dict.items():
        if len(value["word"]) > 1:
            result["phone"].append(key)
            result["word"].append(value["word"])

    result_fin = pd.DataFrame(result)
    # 移除特定条件的行
    index_to_drop = [idx for idx, row in result_fin.iterrows() if row["phone"].endswith("Z") or row["phone"].endswith("S") or len(row["word"]) > 3]
    result_fin.drop(index_to_drop, inplace=True)
    result_fin = result_fin.reset_index(drop=True)
    result_fin.drop_duplicates(subset="word").to_csv(args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and filter phone and word data.")
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("output_file", type=str, help="Path to the output file.")
    args = parser.parse_args()
    main(args)