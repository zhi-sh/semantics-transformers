# -*- coding: utf-8 -*-
# @DateTime :2021/3/3
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
from typing import List, Dict, Type
import torch
import transformers
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import trange, tqdm
from semantics_transformers import SemanticsTransformer
from semantics_transformers.tools import tools, utils


class CrossEncoder(nn.Module):
    def __init__(self,
                 semantics_model: SemanticsTransformer,
                 num_labels: int,
                 device: str = None):
        super(CrossEncoder, self).__init__()
        self.semantics_model = semantics_model
        self.num_labels = num_labels
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._target_device = torch.device(device)

        self.cos_score = nn.Identity()
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, features):
        u, v = [self.semantics_model(feature)['semantics_embedding'] for feature in features]
        output = self.cos_score(torch.cosine_similarity(u, v))
        return output

    def fit(self,
            train_dataloader: DataLoader,
            evaluator=None,
            epochs: int = 1,
            optimizer_class: Type[torch.optim.Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1
            ):
        self.to(self._target_device)
        self.best_score = -1e9

        tools.ensure_path_exist(output_path)

        train_dataloader.collate_fn = self.batching_collate
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        for epoch in trange(epochs, desc='Epoch'):
            training_steps = 0
            self.zero_grad()
            self.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
                outputs = self.forward(features)
                loss_value = self.loss_fct(outputs, labels.float().view(-1))
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    self.zero_grad()
                    self.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

    def predict(self,
                dataloader: DataLoader,
                activation_fct=None,
                need_labels=False,
                ):
        self.to(self._target_device)
        self.eval()

        dataloader.collate_fn = self.batching_collate
        if activation_fct is None:
            activation_fct = nn.Sigmoid() if self.num_labels == 1 else nn.Identity()

        pred_scores = []
        target_labels = []
        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc="Iteration", smoothing=0.05):
                outputs = self.forward(features)
                logits = activation_fct(outputs)
                pred_scores.extend(logits)
                target_labels.extend(labels)
        pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
        if need_labels:
            target_labels = np.asarray(target_labels)
            return pred_scores, target_labels
        return pred_scores

    def batching_collate(self, batch):
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)
        labels = torch.tensor(labels).to(self._target_device)

        features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            utils.batch_to_device(tokenized, self._target_device)
            features.append(tokenized)

        return features, labels

    def tokenize(self, text: str):
        return self._first_module().tokenize(text)

    def save(self, output_path: str):
        saved_model_path = os.path.join(output_path, 'cross_encoder.pth')
        torch.save(self, saved_model_path)

    # --------------------------------------------- 模型属性 ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            # TODO nn.DataParaParallel compatibility
            return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # --------------------------------------------- 内部函数 ----------------------------------------------------------
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps):
        if evaluator is not None:  # 如果存在评估实体，则进行评估模型的过程
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if score >= self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def _first_module(self):
        r'''returns the first moudle of this sequential embedder'''
        return self._modules[next(iter(self._modules))]
