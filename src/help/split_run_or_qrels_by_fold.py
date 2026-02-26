import json
import sys
import argparse
from typing import List, Dict, Set
import os


def read_folds(fold_file: str) -> Dict[str, Dict[str, List[str]]]:
    with open(fold_file, 'r') as f:
        return json.load(f)


def get_queries(fold_number: str, data: Dict[str, Dict[str, List[str]]], split: str) -> List[str]:
    """Get queries for a specific split (training, validation, or testing)"""
    return data[fold_number][split]


def read_run_file(run_file_path: str) -> Dict[str, Set[str]]:
    run_dict: Dict[str, Set[str]] = {}
    with open(run_file_path, 'r') as f:
        for line in f:
            query_id: str = line.split(" ")[0]
            query_run_strings_list: Set[str] = run_dict[query_id] if query_id in run_dict.keys() else set()
            query_run_strings_list.add(line)
            run_dict[query_id] = query_run_strings_list

    return run_dict


def write_to_file(file_path: str, data: List[Set[str]]) -> None:
    with open(file_path, 'a') as f:
        for run_file_strings in data:
            for run_string in run_file_strings:
                f.write("%s\n" % run_string.strip())


def determine_file_type(file_path: str, file_type: str = None) -> str:
    """
    Determine if the file is a qrels or run file

    Args:
        file_path: Path to the input file
        file_type: Explicitly specified file type ('qrels' or 'run')

    Returns:
        'qrels' or 'run'
    """
    if file_type:
        return file_type

    # Check filename for hints
    filename = os.path.basename(file_path).lower()
    if 'qrel' in filename or 'qrels' in filename:
        return 'qrels'
    elif 'run' in filename:
        return 'run'

    # Default to run if unclear
    return 'run'


def get_file_extension(file_type: str) -> str:
    """Get the appropriate file extension based on file type"""
    if file_type == 'qrels':
        return '.qrels.txt'
    else:
        return '.run.txt'


def create_folds_for_run(
        fold_queries: Dict[str, Dict[str, List[str]]],
        run_dict: Dict[str, Set[str]],
        save_dir: str,
        splits_to_process: List[str],
        file_extension: str
) -> None:
    """
    Create fold-wise splits for run/qrels files

    Args:
        fold_queries: Dictionary containing fold information
        run_dict: Dictionary mapping query IDs to run file strings
        save_dir: Directory to save the split files
        splits_to_process: List of splits to process (e.g., ['training', 'validation', 'testing'])
        file_extension: File extension to use (e.g., '.run.txt' or '.qrels.txt')
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

            # Filter the run file to have only those queries which are in this split
            run_fold_list: List[Set[str]] = [
                run_dict[query_id] for query_id in run_dict.keys() if query_id in queries_for_fold
            ]

            # Save to file with appropriate extension
            file_path: str = f"{save_dir}/fold-{fold_num}/{split}{file_extension}"
            write_to_file(file_path=file_path, data=run_fold_list)

            # Track statistics
            split_stats[split] = len(run_fold_list)

        # Print statistics for this fold
        stats_str = ', '.join([f"{split}: {count} queries" for split, count in split_stats.items()])
        print(f'Done Fold-{fold_num} ({stats_str})')


def main():
    parser = argparse.ArgumentParser("Split a run/qrels file into train/val/test files.")
    parser.add_argument("--folds", help='File containing the fold queries.', required=True)
    parser.add_argument("--file", help='Source file to split.', required=True)
    parser.add_argument("--save", help='Directory where data will be saved.', required=True)
    parser.add_argument('--split',
                        help='Which split to process: training, validation, or testing. If not specified, all splits are processed.',
                        choices=['training', 'validation', 'testing'],
                        default=None)
    parser.add_argument('--type',
                        help='File type: qrels or run. If not specified, will be inferred from filename.',
                        choices=['qrels', 'run'],
                        default=None)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading fold queries...')
    fold_query_dict: Dict[str, Dict[str, List[str]]] = read_folds(args.folds)
    print('[Done].')

    print('Loading file...')
    run_dict: Dict[str, Set[str]] = read_run_file(args.file)
    print('[Done].')

    # Determine file type and extension
    file_type = determine_file_type(args.file, args.type)
    file_extension = get_file_extension(file_type)
    print(f'Detected file type: {file_type} (will save as *{file_extension})')

    # Determine which splits to process
    if args.split:
        splits_to_process = [args.split]
    else:
        splits_to_process = ['training', 'validation', 'testing']

    print(f'Creating folds from file for splits: {", ".join(splits_to_process)}...')
    create_folds_for_run(
        fold_queries=fold_query_dict,
        run_dict=run_dict,
        save_dir=args.save,
        splits_to_process=splits_to_process,
        file_extension=file_extension
    )
    print('[Done].')


if __name__ == '__main__':
    main()