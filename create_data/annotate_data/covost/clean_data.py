import os
import json
import shutil
import datasets
import argparse

def copy_audio_files(json_file, target_folder):
    """Copy audio files listed in a JSONL file to a specified target folder."""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            audio_path = item['audio']  # Assuming the path is stored in the 'audio' key
            if os.path.exists(audio_path):
                # Build the target path and copy the file
                shutil.copy(audio_path, os.path.join(target_folder, os.path.basename(audio_path)))
            else:
                print(f"File not found: {audio_path}")

def main(args):
    # Load dataset from disk
    raw_dataset = datasets.load_from_disk(args.dataset_path)
    
    # Open a file for writing
    with open(args.output_jsonl, 'w', encoding='utf-8') as file:
        for item in raw_dataset["train"]:
            # Create a dictionary containing 'audio' and 'text' keys
            jsonl_item = {
                "audio": item['file'],
                "sentence": item['sentence'],
                "translation": item['translation'],
            }
            # Convert the dictionary to a JSON string and write to file
            json.dump(jsonl_item, file)
            file.write('\n')  # Add a newline to separate entries

    # Copy audio files for the specified dataset
    copy_audio_files(args.output_jsonl, args.audio_output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio and text data.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset stored on disk')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Output path for the JSONL file')
    parser.add_argument('--audio_output_folder', type=str, required=True, help='Output folder for copied audio files')

    args = parser.parse_args()
    main(args)
