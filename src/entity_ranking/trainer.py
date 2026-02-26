import torch
import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, data_loader, device, task='classification'):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._data_loader = data_loader
        self._device = device
        self._task = task.lower()

    def make_train_step(self):
        # Builds function that performs a step in the train loop
        def train_step(train_batch):
            # Sets model to TRAIN mode
            self._model.train()

            # Zero the gradients
            self._optimizer.zero_grad()

            if self._task == 'classification':
                # Pointwise: single query-document pair
                batch_score, _, _ = self._model(
                    input_ids=train_batch['input_ids'].to(self._device),
                    input_mask=train_batch['attention_mask'].to(self._device),
                    segment_ids=train_batch['token_type_ids'].to(self._device) if train_batch[
                                                                                      'token_type_ids'] is not None else None,
                )

                # MonoBert outputs 2 classes, convert labels to long
                labels = train_batch['label'].long().to(self._device)
                batch_loss = self._criterion(batch_score, labels)

            elif self._task == 'ranking':
                # Pairwise: query + positive doc vs query + negative doc
                batch_score_pos, _, _ = self._model(
                    input_ids=train_batch['input_ids_pos'].to(self._device),
                    input_mask=train_batch['attention_mask_pos'].to(self._device),
                    segment_ids=train_batch['token_type_ids_pos'].to(self._device) if train_batch[
                                                                                          'token_type_ids_pos'] is not None else None,
                )

                batch_score_neg, _, _ = self._model(
                    input_ids=train_batch['input_ids_neg'].to(self._device),
                    input_mask=train_batch['attention_mask_neg'].to(self._device),
                    segment_ids=train_batch['token_type_ids_neg'].to(self._device) if train_batch[
                                                                                          'token_type_ids_neg'] is not None else None,
                )

                # DuoBert ranking loss: positive should score higher than negative
                batch_loss = self._criterion(
                    batch_score_pos.tanh(),
                    batch_score_neg.tanh(),
                    torch.ones(batch_score_pos.size()).to(self._device)
                )
            else:
                raise ValueError('Task must be `classification` or `ranking`.')

            # Computes gradients
            batch_loss.backward()

            # Updates parameters
            self._optimizer.step()
            self._scheduler.step()

            # Returns the loss
            return batch_loss.item()

        # Returns the function that will be called inside the train loop
        return train_step

    def train(self):
        train_step = self.make_train_step()
        epoch_loss = 0
        num_batch = len(self._data_loader)

        for _, batch in tqdm.tqdm(enumerate(self._data_loader), total=num_batch):
            batch_loss = train_step(batch)
            epoch_loss += batch_loss

        return epoch_loss