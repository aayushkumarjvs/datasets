# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""The Multi-Genre NLI Corpus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six

import tensorflow as tf
from tensorflow_datasets.core import api_utils
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@InProceedings{N18-1101,
  author = "Williams, Adina
            and Nangia, Nikita
            and Bowman, Samuel",
  title = "A Broad-Coverage Challenge Corpus for
           Sentence Understanding through Inference",
  booktitle = "Proceedings of the 2018 Conference of
               the North American Chapter of the
               Association for Computational Linguistics:
               Human Language Technologies, Volume 1 (Long
               Papers)",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  pages = "1112--1122",
  location = "New Orleans, Louisiana",
  url = "http://aclweb.org/anthology/N18-1101"
}
"""

_DESCRIPTION = """\
The Multi-Genre Natural Language Inference (MultiNLI) corpus is a
crowd-sourced collection of 433k sentence pairs annotated with textual
entailment information. The corpus is modeled on the SNLI corpus, but differs in
that covers a range of genres of spoken and written text, and supports a
distinctive cross-genre generalization evaluation. The corpus served as the
basis for the shared task of the RepEval 2017 Workshop at EMNLP in Copenhagen.
"""


class MultiNLIConfig(tfds.core.BuilderConfig):
  """BuilderConfig for MultiNLI."""

  @api_utils.disallow_positional_args
  def __init__(self, text_encoder_config=None, **kwargs):
    """BuilderConfig for MultiNLI.

    Args:
      text_encoder_config: `tfds.features.text.TextEncoderConfig`, configuration
        for the `tfds.features.text.TextEncoder` used for the features feature.
      **kwargs: keyword arguments forwarded to super.
    """
    super(MultiNLIConfig, self).__init__(**kwargs)
    self.text_encoder_config = (
        text_encoder_config or tfds.features.text.TextEncoderConfig())


class MultiNLI(tfds.core.GeneratorBasedBuilder):
  """MultiNLI: The Stanford Question Answering Dataset. Version 1.1."""

  BUILDER_CONFIGS = [
      MultiNLIConfig(
          name="plain_text",
          version="0.0.1",
          description="Plain text",
      ),
      MultiNLIConfig(
          name="bytes",
          version="0.0.1",
          description=("Uses byte-level text encoding with "
                       "`tfds.features.text.ByteTextEncoder`"),
          text_encoder_config=tfds.features.text.TextEncoderConfig(
              encoder=tfds.features.text.ByteTextEncoder()),
      ),
      MultiNLIConfig(
          name="subwords8k",
          version="0.0.1",
          description=("Uses `tfds.features.text.SubwordTextEncoder` with 8k "
                       "vocab size"),
          text_encoder_config=tfds.features.text.TextEncoderConfig(
              encoder_cls=tfds.features.text.SubwordTextEncoder,
              vocab_size=2**13),
      ),
      MultiNLIConfig(
          name="subwords32k",
          version="0.0.2",
          description=("Uses `tfds.features.text.SubwordTextEncoder` with "
                       "32k vocab size"),
          text_encoder_config=tfds.features.text.TextEncoderConfig(
              encoder_cls=tfds.features.text.SubwordTextEncoder,
              vocab_size=2**15),
      ),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "premise":
                tfds.features.Text(
                    encoder_config=self.builder_config.text_encoder_config),
            "hypothesis":
                tfds.features.Text(
                    encoder_config=self.builder_config.text_encoder_config),
            "label":
                tfds.features.Text(
                    encoder_config=self.builder_config.text_encoder_config),
        }),
        # No default supervised_keys (as we have to pass both question
        # and context as input).
        supervised_keys=None,
        urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
        citation=_CITATION,
    )

  def _vocab_text_gen(self, filepath):
    for ex in self._generate_examples(filepath):
      yield " ".join([ex["premise"], ex["hypothesis"], ex["label"]])

  def _split_generators(self, dl_manager):

    # Link to data from GLUE: https://gluebenchmark.com/tasks
    url = ("https://firebasestorage.googleapis.com/v0/b/"
           "mtl-sentence-representations.appspot.com/o/"
           "data%2FMNLI.zip?alt=media&token=50329ea1-e339-"
           "40e2-809c-10c40afff3ce")
    urls_to_download = {
        "mnli": url,
    }
    downloaded_files = dl_manager.download_and_extract(urls_to_download)
    print("downloaded_files=%s" % downloaded_files,)
    mnli_path = os.path.join(downloaded_files["mnli"], "MNLI")
    train_path = os.path.join(mnli_path, "train.tsv")
    # Using dev matched as the default for eval. Can also switch this to
    # dev_mismatched.tsv
    validation_path = os.path.join(mnli_path, "dev_matched.tsv")

    # Generate shared vocabulary
    # maybe_build_from_corpus uses SubwordTextEncoder if that's configured
    self.info.features["premise"].maybe_build_from_corpus(
        self._vocab_text_gen(train_path))
    encoder = self.info.features["premise"].encoder
    # Use maybe_set_encoder because the encoder may have been restored from
    # package data.
    self.info.features["premise"].maybe_set_encoder(encoder)
    self.info.features["hypothesis"].maybe_set_encoder(encoder)
    self.info.features["label"].maybe_set_encoder(encoder)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,
            gen_kwargs={"filepath": train_path}),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=1,
            gen_kwargs={"filepath": validation_path}),
    ]

  def _generate_examples(self, filepath):
    """Generate mnli examples.

    Args:
      filepath: a string
    Yields:
      dictionaries containing "premise", "hypothesis" and "label" strings
    """
    for idx, line in enumerate(tf.gfile.Open(filepath, "rb")):
      if idx == 0: continue  # skip header
      line = _to_unicode_utf8(line.strip())
      split_line = line.split("\t")
      # Works for both splits even though dev has some extra human labels.
      yield {
          "premise": split_line[8],
          "hypothesis": split_line[9],
          "label": split_line[-1]
      }


def _to_unicode_utf8(s):
  return unicode(s, "utf-8") if six.PY2 else s.decode("utf-8")
