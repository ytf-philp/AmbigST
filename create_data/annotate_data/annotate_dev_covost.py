import pandas as pd
import csv
import argparse

def load_data(file_path):
    samples = []
    with open(file_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples.append([dict(e) for e in reader])
    return pd.DataFrame(samples[0])

def main(args):
    # Load the data
    train_df = load_data(args.input_file)
    
    # Initialize columns with default values
    train_df["audio_align"] = -1
    train_df["text_align"] = -1
    
    # Save the dataframe to a new TSV file
    train_df.to_csv(args.output_file, sep="\t", index=None, quoting=csv.QUOTE_NONE)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process file paths for data handling.')

    # Add arguments
    parser.add_argument('input_file', type=str, help='Path to the input TSV file')
    parser.add_argument('output_file', type=str, help='Path to the output TSV file')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
