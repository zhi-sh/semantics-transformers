# -*- coding: utf-8 -*-
# @DateTime :2021/3/2
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import csv, logging, os
from typing import List
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from semantics_transformers.tools import utils
from semantics_transformers.evaluations import AbstractEvaluation

logger = logging.getLogger(__name__)


class LabelAccuracyEvaluator(AbstractEvaluation):
    def __init__(self, dataloader: DataLoader, name: str = '', softmax_model=None):
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        self.csv_file = f'accuracy_evaluation_{name}_results.csv'
        self.csv_headers = ['epoch', 'steps', 'accuracy']

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        # 构造日志输出格式
        if epoch == -1:
            if steps == -1:
                out_txt = "after epoch {}:".format(epoch)
            else:
                out_txt = "in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ':'
        logger.info(f"evaluation on the {self.name} dataset {out_txt}")

        self.dataloader.collate_fn = model.batching_collate
        for step, batch in enumerate(tqdm(self.dataloader, desc='Evaluating')):
            features, label_ids = batch
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)
            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids.view(-1)).sum().item()
        accuracy = correct / total
        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        # 评估结果保存至文件
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode='w', encoding='utf-8') as fout:
                    writer = csv.writer(fout)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, mode='a', encoding='utf-8') as fout:
                    writer = csv.writer(fout)
                    writer.writerow([epoch, steps, accuracy])
        print("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        return accuracy
