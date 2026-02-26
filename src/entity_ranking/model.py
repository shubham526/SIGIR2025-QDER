import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Tuple, Dict, List, Any


class Bert(nn.Module):
    def __init__(self,
                 pretrained: str,
                 mode: str = 'cls') -> None:

        super(Bert, self).__init__()
        self._pretrained = pretrained
        self._mode = mode

        self._config = AutoConfig.from_pretrained(self._pretrained, output_hidden_states=True)
        self._model = AutoModel.from_pretrained(self._pretrained, config=self._config)

    def forward(self,
                input_ids: torch.Tensor,
                input_mask: torch.Tensor = None,
                segment_ids: torch.Tensor = None) -> Tuple[Any, Any, Any]:

        output = self._model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        if self._mode == 'cls':
            logits = output[0][:, 0, :]
        elif self._mode == 'pooling':
            logits = output[1]
        else:
            raise ValueError('Mode must be `cls` or `pooling`.')
        score = self._dense(logits).squeeze(-1)
        return score, logits, output.hidden_states


class MonoBert(Bert):
    def __init__(self,
                 pretrained: str,
                 mode: str = 'cls') -> None:
        Bert.__init__(self, pretrained, mode)
        self._dense = nn.Linear(self._config.hidden_size, 2)


class DuoBert(Bert):
    def __init__(self,
                 pretrained: str,
                 mode: str = 'cls') -> None:
        Bert.__init__(self, pretrained, mode)
        self._dense = nn.Linear(self._config.hidden_size, 1)