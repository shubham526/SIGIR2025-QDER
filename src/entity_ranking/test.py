import torch
import utils
import argparse
from dataloader import EntityRankingDataLoader
from model import MonoBert, DuoBert
from transformers import AutoTokenizer
from dataset import EntityRankingDataset


def main():
    parser = argparse.ArgumentParser("Script to test a model.")
    parser.add_argument('--test', help='Test data.', required=True, type=str)
    parser.add_argument('--run', help='Test run file.', required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--query-enc',
                        help='Name of model (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5). '
                             'Default: bert.', type=str, default='bert')
    parser.add_argument('--task',
                        help='Task type: classification (MonoBert) or ranking (DuoBert). Default: classification',
                        type=str, default='classification', choices=['classification', 'ranking'])
    parser.add_argument('--mode', help='Pooling mode (cls|pooling). Default: cls', type=str, default='cls')
    parser.add_argument('--checkpoint', help='Name of checkpoint to load.', required=True, type=str)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 0', type=int,
                        default=0)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    model_map = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta': 'roberta-base',
        'deberta': 'microsoft/deberta-base',
        'ernie': 'nghuyong/ernie-2.0-base-en',
        'electra': 'google/electra-small-discriminator',
        'conv-bert': 'YituTech/conv-bert-base',
        't5': 't5-base'
    }

    pretrain = vocab = model_map[args.query_enc]
    tokenizer = AutoTokenizer.from_pretrained(vocab, model_max_length=args.max_len)
    print('Query Encoder: {}'.format(args.query_enc))
    print('Task: {}'.format(args.task))
    print('Pooling Mode: {}'.format(args.mode))

    print('Reading test data...')
    test_set = EntityRankingDataset(
        dataset=args.test,
        tokenizer=tokenizer,
        max_len=args.max_len,
        train=False,
        task=args.task
    )
    print('[Done].')

    print('Creating data loader...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))

    test_loader = EntityRankingDataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print('[Done].')

    # Choose model based on task
    if args.task == 'classification':
        model = MonoBert(pretrained=pretrain, mode=args.mode)
        print('Using MonoBert for classification task')
    else:  # ranking
        model = DuoBert(pretrained=pretrain, mode=args.mode)
        print('Using DuoBert for ranking task')

    print('Loading checkpoint...')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print('[Done].')

    print('Using device: {}'.format(device))
    model.to(device)

    print("Starting to test...")

    res_dict = utils.evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        task=args.task
    )

    print('Writing run file...')
    utils.save_trec(args.run, res_dict)
    print('[Done].')

    print('[Done].')
    print('Run file saved to ==> {}'.format(args.run))


if __name__ == '__main__':
    main()