import argparse
import datasets
import soundfile as sf

def process_dataset(batch):
    audio_path = batch["file"]
    try:
        info = sf.info(audio_path)
        is_readable = True
    except Exception:
        is_readable = False

    batch["path"] = audio_path
    batch["sentence"] = batch['sentence'],
    batch["translation"] = batch['translation']
    batch["is_readable"] = is_readable
    return batch

def is_readable(flag):
    return flag

def main(args):
    raw_dataset = datasets.load_dataset(
        args.script_path,
        'fr_en',
        data_dir=args.data_dir,
        split={"train": 'train', "test": 'test', "validation": 'validation'}
    )

    columns_to_remove = raw_dataset["train"].column_names
    dataset = raw_dataset.map(
        "./AmbigST/create_data/annotate_data/covost/covost.py",
        load_from_cache_file=False,
        num_proc=8,
        remove_columns=["audio"]
    )

    dataset = dataset.filter(
        is_readable,
        input_columns=["is_readable"]
    )

    dataset.save_to_disk(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset to check audio file readability and save processed data.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where the data is stored')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the processed dataset')

    args = parser.parse_args()
    main(args)
