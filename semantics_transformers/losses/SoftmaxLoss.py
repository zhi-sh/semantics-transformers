# -*- coding: utf-8 -*-
# @DateTime :2021/3/2
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging
from typing import Dict, Iterable
import torch
from torch import nn, Tensor
from semantics_transformers import SemanticsTransformer

logger = logging.getLogger(__name__)


class SoftmaxLoss(nn.Module):
    def __init__(self,
                 model: SemanticsTransformer,
                 embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False
                 ):
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logger.info(f"Softmax loss : #Vecotrs concatenated: {num_vectors_concatenated}")
        print(num_vectors_concatenated * embedding_dimension)
        self.classifier = nn.Linear(num_vectors_concatenated * embedding_dimension, num_labels)

    def forward(self, features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(feature)['semantics_embedding'] for feature in features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)
        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))
        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a - rep_b)

        embeddings = torch.cat(vectors_concat, 1)
        output = self.classifier(embeddings)
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output
