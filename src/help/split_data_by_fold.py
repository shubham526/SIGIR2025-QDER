import json
import sys
from tqdm import tqdm
import argparse
from typing import List, Dict, Any


def read_folds(fold_file: str) -> Dict[str, Dict[str, List[str]]]:
    with open(fold_file, 'r') as f:
        return json.load(f)


def read_data(data: str):
    with open(data, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]


def get_queries(fold_number: str, data: Dict[str, Dict[str, List[str]]], split: str) -> List[str]:
    """Get queries for a specific split (training, validation, or testing)"""
    return data[fold_number][split]


def write_to_file(file_path: str, data: List[str]) -> None:
    with open(file_path, 'a') as f:
        for d in data:
            f.write('%s\n' % d)


def create_fold_data(
        fold_data_dict: Dict[str, Dict[str, List[str]]],
        data: List[Dict[str, Any]],
        save_dir: str,
        split: str,
) -> None:
    """
    Create fold data for a specific split (training, validation, or testing)

    Args:
        fold_data_dict: Dictionary containing fold information
        data: List of data samples
        save_dir: Directory to save the split data
        split: One of 'training', 'validation', or 'testing'
    """
    for fold_num in range(5):
        # Get queries for this fold and split
        fold_queries: List[str] = get_queries(fold_number=str(fold_num), data=fold_data_dict, split=split)

        # Filter data for this fold's queries
        fold_data: List[str] = [json.dumps(d) for d in data if d['query_id'] in fold_queries]

        # Save to appropriate file
        save_file_path: str = f"{save_dir}/fold-{fold_num}/{split}.jsonl"
        write_to_file(file_path=save_file_path, data=fold_data)

        print(f'Done Fold-{fold_num} {split}: {len(fold_data)} samples')


def main():
    parser = argparse.ArgumentParser("Split data by fold for k-fold CV with train/val/test splits.")
    parser.add_argument("--folds", help='Path to file containing the fold queries.', required=True)
    parser.add_argument("--data", help='Path to data file.', required=True)
    parser.add_argument("--save", help='Path to directory where data will be saved.', required=True)
    parser.add_argument('--split',
                        help='Which split to process: training, validation, or testing. If not specified, all splits are processed.',
                        choices=['training', 'validation', 'testing'],
                        default=None)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading fold queries...')
    fold_query_dict: Dict[str, Dict[str, List[str]]] = read_folds(args.folds)
    print('[Done].')

    print('Loading data...')
    data: List[Dict[str, Any]] = read_data(args.data)
    print('[Done].')

    # Process specified split or all splits
    if args.split:
        splits_to_process = [args.split]
    else:
        splits_to_process = ['training', 'validation', 'testing']

    for split in splits_to_process:
        print(f'\nCreating fold-wise {split} data...')
        create_fold_data(
            fold_data_dict=fold_query_dict,
            data=data,
            save_dir=args.save,
            split=split,
        )
        print(f'[Done with {split}].')


if __name__ == '__main__':
    main()