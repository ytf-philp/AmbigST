import argparse
import json
import textgrid
import tqdm

def convert(path):
    """Convert TextGrid file to word times and texts."""
    tg = textgrid.TextGrid.fromFile(path)
    word_time = [tg[0][j].maxTime for j in range(len(tg[0]))]
    word_text = [tg[0][j].mark for j in range(len(tg[0]))]
    word_time = '*'.join(map(str, word_time))
    word_text = '*'.join(word_text)
    return word_time, word_text

def process_sentence(sentence):
    """Encode the sentence in UTF-8 format."""
    return sentence.encode('utf-8').decode('utf-8')

def replace(output, origin):
    """Replace placeholders in the output with words from the original sentence."""
    output_list = output.split(',')
    origin_list = origin.split(' ')
    output_index = [idx for idx, word in enumerate(output_list) if word != '']
    if len(output_index) != len(origin_list):
        return None
    for idx, word in enumerate(origin_list):
        output_list[output_index[idx]] = word
    return ','.join(output_list)

def process_files(input_file, output_file, text_grid):
    """Process each line in the input file and write results to the output file."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            item = json.loads(line)
            item['sentence'] = process_sentence(item['sentence'][0])
            path = item["audio"].split("/")[-1][:-4]
            try:
                word_time, word_text = convert(text_grid + path + ".TextGrid")
            except Exception as e:
                print(f"Failed to process file {path}: {str(e)}")
                continue
            item['word_time'] = word_time
            item['word_text'] = word_text
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Process and convert audio data files.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument('--text_grid_folder', type=str, required=True, help="Directory containing TextGrid files")

    args = parser.parse_args()

    process_files(args.input_file, args.output_file, args.text_grid_folder)

if __name__ == "__main__":
    main()


def main():
    text_grid = "/workspace/AmbigST-de/data_process/de/train_align/"
    process_files('/workspace/AmbigST-de/data/de/dataset_train.jsonl', 
                  '/workspace/AmbigST-de/data/de/dataset_train_raw.jsonl', text_grid)

if __name__ == "__main__":
    main()
