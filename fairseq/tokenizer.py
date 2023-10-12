# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):  # 替换字符串中"\s+"->" ",去除首尾空白并切片
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()
