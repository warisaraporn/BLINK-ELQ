# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import prettytable

import blink.main_dense as main_dense
import blink.candidate_ranking.utils as utils

DATASETS = [
    {
        "name": "zeshel",
        "filename": "data/zeshel/blink_format/test.jsonl",
    },
]

PARAMETERS = {
    "faiss_index": None,
    "index_path": None,
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "entity_catalogue": "models/entity.jsonl",
    "entity_encoding": "models/all_entities_large.t7",
    "biencoder_model": "models/zeshel/biencoder/pytorch_model.bin",
    "biencoder_config": "models/zeshel/biencoder/training_params.txt",
    "crossencoder_model": "models/crossencoder_wiki_large.bin",
    "crossencoder_config": "models/crossencoder_wiki_large.json",
    "output_path": "output",
    "fast": False,
    "top_k": 100,
}
args = argparse.Namespace(**PARAMETERS)

logger = utils.get_logger(args.output_path)

models = main_dense.load_models(args, logger)

table = prettytable.PrettyTable(
    [
        "DATASET",
        "biencoder accuracy",
        "recall at 100",
        "crossencoder normalized accuracy",
        "overall unormalized accuracy",
        "support",
    ]
)

for dataset in DATASETS:
    logger.info(dataset["name"])
    PARAMETERS["test_mentions"] = dataset["filename"]

    args = argparse.Namespace(**PARAMETERS)
    (
        biencoder_accuracy,
        recall_at,
        crossencoder_normalized_accuracy,
        overall_unormalized_accuracy,
        num_datapoints,
        predictions,
        scores,
    ) = main_dense.run(args, logger, *models)

    table.add_row(
        [
            dataset["name"],
            round(biencoder_accuracy, 4),
            round(recall_at, 4),
            round(crossencoder_normalized_accuracy, 4),
            round(overall_unormalized_accuracy, 4),
            num_datapoints,
        ]
    )

logger.info("\n{}".format(table))
