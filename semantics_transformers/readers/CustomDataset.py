# -*- coding: utf-8 -*-
# @DateTime :2021/3/2
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import pandas as pd
from semantics_transformers.readers import InputExample


class CustomDataset:
    def __init__(self, dataset_folder: str, label2idx: dict):
        self.dataset_folder = dataset_folder
        self.label2idx = label2idx  # 如果过多，可以通过JSON字典的方式加载

    def get_examples(self, filename, header, max_examples=0, has_label: bool = True, delimiter=','):
        data = pd.read_csv(os.path.join(self.dataset_folder, filename), delimiter=delimiter, header=header)

        examples = []
        for idx, row in data.iterrows():
            if has_label:
                examples.append(InputExample(guid=str(idx), texts=[row[0], row[1]], label=self.map_labels(row[2])))
            else:
                examples.append(InputExample(guid=str(idx), texts=[row[0], row[1]]))

            # 若设置max_examples，则读取固定数量的样本数据
            if 0 < max_examples <= len(examples):
                break

        return examples

    def map_labels(self, label):
        if not isinstance(label, str):
            label = str(label)
        return self.label2idx[label.strip().lower()]

    @property
    def num_labels(self):
        return len(self.label2idx)
