# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""The General Language Understanding Evaluation (GLUE) benchmark."""


import json
import textwrap

import datasets


_XGLUE_CITATION = """\
@article{Liang2020XGLUEAN,
  title={XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation},
  author={Yaobo Liang and Nan Duan and Yeyun Gong and Ning Wu and Fenfei Guo and Weizhen Qi
  and Ming Gong and Linjun Shou and Daxin Jiang and Guihong Cao and Xiaodong Fan and Ruofei
  Zhang and Rahul Agrawal and Edward Cui and Sining Wei and Taroon Bharti and Ying Qiao
  and Jiun-Hung Chen and Winnie Wu and Shuguang Liu and Fan Yang and Daniel Campos
  and Rangan Majumder and Ming Zhou},
  journal={arXiv},
  year={2020},
  volume={abs/2004.01401}
}
"""

_XGLUE_DESCRIPTION = """\
XGLUE is a new benchmark dataset to evaluate the performance of cross-lingual pre-trained
models with respect to cross-lingual natural language understanding and generation.
The benchmark is composed of the following 11 tasks:
- NER
- POS Tagging (POS)
- News Classification (NC)
- MLQA
- XNLI
- PAWS-X
- Query-Ad Matching (QADSM)
- Web Page Ranking (WPR)
- QA Matching (QAM)
- Question Generation (QG)
- News Title Generation (NTG)
For more information, please take a look at https://microsoft.github.io/XGLUE/.
"""

_XGLUE_ALL_DATA = "https://xglue.blob.core.windows.net/xglue/xglue_full_dataset.tar.gz"

_LANGUAGES = {
    "qadsm": ["en"],
}

_PATHS = {
    "qadsm": {
        # "train": "unlabeled.qadsm.train.txt",
        "train": "prediction.tsv",
    }
}


class XGlueConfig(datasets.BuilderConfig):
    """BuilderConfig for XGLUE."""

    def __init__(
        self,
        data_dir,
        **kwargs,
    ):
        """BuilderConfig for XGLUE.
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(XGlueConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.data_dir = data_dir


class XGlue(datasets.GeneratorBasedBuilder):
    """The Cross-lingual Pre-training, Understanding and Generation (XGlue) Benchmark."""

    BUILDER_CONFIGS = [
        XGlueConfig(
            name="qadsm",
            data_dir="QADSM"
        )
    ]

    def _info(self):
        if self.config.name == "qadsm":
            features = {
                "query": datasets.Value("string"),
                "ad_title": datasets.Value("string"),
                "ad_description": datasets.Value("string"),
                "relevance_label": datasets.Value("float"),
            }

        return datasets.DatasetInfo(
            description=_XGLUE_DESCRIPTION,
            features=datasets.Features(features),
            homepage='',
            citation='',
        )

    def _split_generators(self, dl_manager):
        # archive = dl_manager.download(_XGLUE_ALL_DATA)
        data_folder = f"data/{self.config.data_dir}"
        name = self.config.name

        languages = _LANGUAGES[name]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": f"{data_folder}/{_PATHS[name]['train']}",
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, data_path, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(data_path, encoding="utf-8") as f:
            for key, line in enumerate(f):
                tokens = line.strip('\n').split('\t')
                # Yields examples as (key, example) tuples
                yield key, {
                    "query": tokens[0],
                    "ad_title": tokens[1],
                    "ad_description": tokens[2],
                    "relevance_label": tokens[3]
                }
