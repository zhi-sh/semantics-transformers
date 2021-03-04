# -*- coding: utf-8 -*-
# @DateTime :2021/3/4
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import csv, logging, os
import numpy as np
from torch.utils.data import DataLoader
from semantics_transformers.evaluations import AbstractEvaluation

logger = logging.getLogger(__name__)


class CEBinaryAccuracyEvaluator(AbstractEvaluation):
    def __init__(self, dataloader: DataLoader, name: str = '', threshold: int = 0.5):
        self.dataloader = dataloader
        self.name = name
        self.threshold = threshold

        self.csv_file = f'ceb_evaluation_{name}_results.csv'
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

        pred_scores, labels = model.predict(self.dataloader, need_labels=True)
        pred_labels = pred_scores > self.threshold

        assert len(pred_labels) == len(labels)
        accuracy = np.sum(pred_labels == labels) / len(labels)
        logger.info(f"Accuracy: {accuracy * 100:.6f}")

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
        print(f"Accuracy: {accuracy:.6f}\n")
        return accuracy
