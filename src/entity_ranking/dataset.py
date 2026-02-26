from typing import Dict, Any
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer


class EntityRankingDataset(Dataset):
    def __init__(
            self,
            dataset,
            max_len,
            tokenizer: AutoTokenizer,
            train: bool,
            task: str = 'classification'  # 'classification' or 'ranking'
    ):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._train = train
        self._task = task
        self._read_data()
        self._max_len = max_len

        self._count = len(self._examples)

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for i, line in enumerate(f)]

    def _create_bert_input(self, text_a, text_b=None):
        """
        Create BERT input for query-document pair or single text.
        If text_b is provided, creates [CLS] text_a [SEP] text_b [SEP]
        """
        # Tokenize the query-document pair
        encoded_dict = self._tokenizer(
            text=text_a,
            text_pair=text_b,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True,  # For segment IDs
        )

        return (encoded_dict['input_ids'],
                encoded_dict['attention_mask'],
                encoded_dict.get('token_type_ids', None))

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        query = example['query'].strip()

        if self._task == 'classification':
            # Pointwise: single query-document pair
            # Support both 'doc_text' and 'doc' field names
            doc_text = example.get('doc_text', example.get('doc', '')).strip()
            input_ids, attention_mask, token_type_ids = self._create_bert_input(query, doc_text)

            if self._train:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': example['label']
                }
            else:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': example['label'],
                    'query_id': example['query_id'],
                    'doc_id': example['doc_id']
                }

        elif self._task == 'ranking':
            # Pairwise: query + positive doc and query + negative doc
            doc_pos_text = example.get('doc_pos_text', example.get('doc_text', '')).strip()
            doc_neg_text = example.get('doc_neg_text', '').strip()

            input_ids_pos, attention_mask_pos, token_type_ids_pos = self._create_bert_input(query, doc_pos_text)
            input_ids_neg, attention_mask_neg, token_type_ids_neg = self._create_bert_input(query, doc_neg_text)

            if self._train:
                return {
                    'input_ids_pos': input_ids_pos,
                    'attention_mask_pos': attention_mask_pos,
                    'token_type_ids_pos': token_type_ids_pos,
                    'input_ids_neg': input_ids_neg,
                    'attention_mask_neg': attention_mask_neg,
                    'token_type_ids_neg': token_type_ids_neg,
                }
            else:
                # For evaluation, we still process one doc at a time
                # Support both 'doc_text' and 'doc' field names
                doc_text = example.get('doc_text', example.get('doc', doc_pos_text)).strip()
                input_ids, attention_mask, token_type_ids = self._create_bert_input(query, doc_text)
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': example.get('label', 1),
                    'query_id': example['query_id'],
                    'doc_id': example['doc_id']
                }

    def collate(self, batch):
        if self._task == 'classification':
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            token_type_ids = torch.tensor([item['token_type_ids'] for item in batch]) if batch[0][
                                                                                             'token_type_ids'] is not None else None
            label = torch.from_numpy(np.array([item['label'] for item in batch])).float()

            if self._train:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': label
                }
            else:
                query_id = [item['query_id'] for item in batch]
                doc_id = [item['doc_id'] for item in batch]
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': label,
                    'query_id': query_id,
                    'doc_id': doc_id
                }

        elif self._task == 'ranking':
            if self._train:
                input_ids_pos = torch.tensor([item['input_ids_pos'] for item in batch])
                attention_mask_pos = torch.tensor([item['attention_mask_pos'] for item in batch])
                token_type_ids_pos = torch.tensor([item['token_type_ids_pos'] for item in batch]) if batch[0][
                                                                                                         'token_type_ids_pos'] is not None else None

                input_ids_neg = torch.tensor([item['input_ids_neg'] for item in batch])
                attention_mask_neg = torch.tensor([item['attention_mask_neg'] for item in batch])
                token_type_ids_neg = torch.tensor([item['token_type_ids_neg'] for item in batch]) if batch[0][
                                                                                                         'token_type_ids_neg'] is not None else None

                return {
                    'input_ids_pos': input_ids_pos,
                    'attention_mask_pos': attention_mask_pos,
                    'token_type_ids_pos': token_type_ids_pos,
                    'input_ids_neg': input_ids_neg,
                    'attention_mask_neg': attention_mask_neg,
                    'token_type_ids_neg': token_type_ids_neg,
                }
            else:
                # Eval mode: same as classification
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                attention_mask = torch.tensor([item['attention_mask'] for item in batch])
                token_type_ids = torch.tensor([item['token_type_ids'] for item in batch]) if batch[0][
                                                                                                 'token_type_ids'] is not None else None
                label = torch.from_numpy(np.array([item['label'] for item in batch])).float()
                query_id = [item['query_id'] for item in batch]
                doc_id = [item['doc_id'] for item in batch]

                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': label,
                    'query_id': query_id,
                    'doc_id': doc_id
                }