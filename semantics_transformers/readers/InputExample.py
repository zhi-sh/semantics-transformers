# -*- coding: utf-8 -*-
# @DateTime :2021/3/2
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from typing import List, Union


class InputExample:
    r'''
        文本对 样本
        {[text1, text2], label}
    '''
    def __init__(self, guid: str = '', texts: List[str] = None, label: Union[int, float] = 0):
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return f"<InputExample> guid: {self.guid}, label: {self.label}, texts: {self.texts}"
