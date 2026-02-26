import json
import sys
import argparse
from typing import List, Dict


def read_folds(fold_file: str) -> Dict[str, Dict[str, List[str]]]:
    with open(fold_file, 'r') as f:
        return json.load(f)


def get_queries(fold_number: str, data: Dict[str, Dict[str, List[str]]], split: str) -> List[str]:
    """Get queries for a specific split (training, validation, or testing)"""
    return data[fold_number][split]


def read_tsv(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        return dict({(line.strip().split('\t')[0], line.strip().split('\t')[1]) for line in f})


def write_to_file(file_path: str, data: Dict[str, str]) -> None:
    with open(file_path, 'a') as f:
        for k, v in data.items():
            f.write("%s\t%s\n" % (k, v))


def create_folds_for_run(
        fold_queries: Dict[str, Dict[str, List[str]]],
        data_dict: Dict[str, str],
        save_dir: str,
        splits_to_process: List[str]
) -> None:
    """
    Create fold-wise splits for TSV files

    Args:
        fold_queries: Dictionary containing fold information
        data_dict: Dictionary mapping query IDs to query text
        save_dir: Directory to save the split files
        splits_to_process: List of splits to process (e.g., ['training', 'validation', 'testing'])
    """
    for fold_num in range(5):
        split_stats = {}

        for split in splits_to_process:
            # Get the queries for this fold and split
            queries_for_fold: List[str] = get_queries(
                fold_number=str(fold_num),
                data=fold_queries,
                split=split
            )

            # Filter the data to have only those queries which are in this split
            split_data: Dict[str, str] = dict({
                (query_id, query) for query_id, query in data_dict.items() if query_id in queries_for_fold
            })

            # Save to file
            file_path: str = f"{save_dir}/fold-{fold_num}/{split}.tsv"
            write_to_file(file_path=file_path, data=split_data)

            # Track statistics
            split_stats[split] = len(split_data)

        # Print statistics for this fold
        stats_str = ', '.join([f"{split}: {count} queries" for split, count in split_stats.items()])
        print(f'Done Fold-{fold_num} ({stats_str})')


def main():
    parser = argparse.ArgumentParser("Split a TSV file into train/val/test files.")
    parser.add_argument("--folds", help='File containing the fold queries.', required=True)
    parser.add_argument("--file", help='Source TSV file to split.', required=True)
    parser.add_argument("--save", help='Directory where data will be saved.', required=True)
    parser.add_argument('--split',
                        help='Which split to process: training, validation, or testing. If not specified, all splits are processed.',
                        choices=['training', 'validation', 'testing'],
                        default=None)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading fold queries...')
    fold_query_dict: Dict[str, Dict[str, List[str]]] = read_folds(args.folds)
    print('[Done].')

    print('Loading file...')
    data_dict: Dict[str, str] = read_tsv(file_path=args.file)
    print('[Done].')

    # Determine which splits to process
    if args.split:
        splits_to_process = [args.split]
    else:
        splits_to_process = ['training', 'validation', 'testing']

    print(f'Creating folds from file for splits: {", ".join(splits_to_process)}...')
    create_folds_for_run(
        fold_queries=fold_query_dict,
        data_dict=data_dict,
        save_dir=args.save,
        splits_to_process=splits_to_process
    )
    print('[Done].')


if __name__ == '__main__':
    main()