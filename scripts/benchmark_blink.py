# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import argparse
import prettytable

import blink.main_dense as main_dense
import blink.candidate_ranking.utils as utils

DATASETS = [
    {
        "name": "mintaka",
        "model": "wikipedia",
        "filename": "/workspace/datasets/blink/mintaka/test.jsonl",
        ### wikipedia_model
        "biencoder_model": "models/biencoder_wiki_large.bin",
        "biencoder_config": "models/biencoder_wiki_large.json",
        "crossencoder_model": "models/crossencoder_wiki_large.bin", 
        "crossencoder_config": "models/crossencoder_wiki_large.json",
        ### ft_models
        # "biencoder_model": "/workspace/BLINK/blink_ft_modelsv2/mintaka_modelv2/biencoder/epoch_1/pytorch_model.bin",
        # "biencoder_config": "/workspace/BLINK/blink_ft_modelsv2/mintaka_modelv2/biencoder/training_params.txt",
        # "crossencoder_model": "/workspace/BLINK/blink_ft_modelsv2/mintaka_modelv2/crossencoder    #/epoch_1/pytorch_model.bin",
        # "crossencoder_config": "/workspace/BLINK/blink_ft_modelsv2/mintaka_modelv2/crossencoder/training_params.txt"
    },
    {
        "name": "webqsp",
        "model": "wikipedia",
        "filename": "/workspace/datasets/blink/webqsp/test.jsonl",
        ### wikipedia_model
        "biencoder_model": "models/biencoder_wiki_large.bin",
        "biencoder_config": "models/biencoder_wiki_large.json",
        "crossencoder_model": "models/crossencoder_wiki_large.bin", 
        "crossencoder_config": "models/crossencoder_wiki_large.json",
        ### ft_models
        # "biencoder_model": "/workspace/BLINK/blink_ft_modelsv2/webqsp_modelv2/biencoder/epoch_0/pytorch_model.bin",
        # "biencoder_config": "/workspace/BLINK/blink_ft_modelsv2/webqsp_modelv2/biencoder/training_params.txt",
        # "crossencoder_model": "/workspace/BLINK/blink_ft_modelsv2/webqsp_modelv2/crossencoder    #/epoch_1/pytorch_model.bin",
        # "crossencoder_config": "/workspace/BLINK/blink_ft_modelsv2/webqsp_modelv2/crossencoder/training_params.txt"
    },
    {
        "name": "graphq",
        "model": "wikipedia",
        "filename": "/workspace/datasets/blink/graphq/test.jsonl",
        ### wikipedia_model
        "biencoder_model": "models/biencoder_wiki_large.bin",
        "biencoder_config": "models/biencoder_wiki_large.json",
        "crossencoder_model": "models/crossencoder_wiki_large.bin", 
        "crossencoder_config": "models/crossencoder_wiki_large.json",
        ### ft_models
        # "biencoder_model": "/workspace/BLINK/blink_ft_modelsv2/graphq_modelv2/biencoder/epoch_0/pytorch_model.bin",
        # "biencoder_config": "/workspace/BLINK/blink_ft_modelsv2/graphq_modelv2/biencoder/training_params.txt",
        # "crossencoder_model": "/workspace/BLINK/blink_ft_modelsv2/graphq_modelv2/crossencoder    #/epoch_1/pytorch_model.bin",
        # "crossencoder_config": "/workspace/BLINK/blink_ft_modelsv2/graphq_modelv2/crossencoder/training_params.txt"
    },
    {
        "name": "shadowlink_top",
        "model": "wikipedia",
        "filename": "/workspace/datasets/blink/shadowlink/top.jsonl",
        ### wikipedia_model
        "biencoder_model": "models/biencoder_wiki_large.bin",
        "biencoder_config": "models/biencoder_wiki_large.json",
        "crossencoder_model": "models/crossencoder_wiki_large.bin", 
        "crossencoder_config": "models/crossencoder_wiki_large.json",
    },
    {
        "name": "shadowlink_tail",
        "model": "wikipedia",
        "filename": "/workspace/datasets/blink/shadowlink/tail.jsonl",
        ### wikipedia_model
        "biencoder_model": "models/biencoder_wiki_large.bin",
        "biencoder_config": "models/biencoder_wiki_large.json",
        "crossencoder_model": "models/crossencoder_wiki_large.bin", 
        "crossencoder_config": "models/crossencoder_wiki_large.json",
    },
    {
        "name": "shadowlink_shadow",
        "model": "wikipedia",
        "filename": "/workspace/datasets/blink/shadowlink/shadow.jsonl",
        ### wikipedia_model
        "biencoder_model": "models/biencoder_wiki_large.bin",
        "biencoder_config": "models/biencoder_wiki_large.json",
        "crossencoder_model": "models/crossencoder_wiki_large.bin", 
        "crossencoder_config": "models/crossencoder_wiki_large.json",
    }
]

raw_data = [
    # "/workspace/datasets/aida_test.json", ### aida
    "/workspace/datasets/mintaka_test_data_el.json", ### mintaka
    "/workspace/datasets/webqsp_test_data_el.json", ### webqsp
    "/workspace/datasets/graphq_test_data_el.json", ### graphq
    "/workspace/datasets/shadowlink_top.json", ### shadow_top
    "/workspace/datasets/shadowlink_tail.json", ### shadow_tail
    "/workspace/datasets/shadowlink_shadow.json", ### shadow_shadow
]

### get number of total entity in dataset
num_raw_data_entity = []
for dt in raw_data:
    data = [json.loads(line) for line in open(dt, 'r', encoding='utf-8')]
    entity_count = 0
    for d in data:
        if 'spans' in d.keys():
            entity_count += len(d['spans'])
        if 'entity_mentions' in d.keys():
            entity_count += len(d['entity_mentions'])
        if 'entity_name' in d.keys():
            entity_count += len(d['entity_name'])
        if 'entity' in d.keys():
            entity_count += len(d['entity'])
    num_raw_data_entity.append(entity_count)

PARAMETERS = {
    "faiss_index": None,
    "index_path": None,
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "entity_catalogue": "models/entity.jsonl",
    "entity_encoding": "models/all_entities_large.t7",
    "output_path": "outputv2",
    "fast": False,
    "top_k": 100,
}

table = prettytable.PrettyTable(
    [
        "DATASET",
        "MODEL",
        "biencoder accuracy",
        "biencoder precision",
        "biencoder recall",
        "biencoder f1",
        "crossencoder normalized accuracy",
        "crossencoder precision",
        "crossencoder recall",
        "crossencoder f1",
        "support",
        "Ratio of in KB entity/All entity"

    ]
)

for i in range(len(DATASETS)):
    PARAMETERS["biencoder_model"] = DATASETS[i]["biencoder_model"]
    PARAMETERS["biencoder_config"] = DATASETS[i]["biencoder_config"]
    PARAMETERS["crossencoder_model"] = DATASETS[i]["crossencoder_model"]
    PARAMETERS["crossencoder_config"] = DATASETS[i]["crossencoder_config"]
    PARAMETERS["test_mentions"] = DATASETS[i]["filename"]

    args = argparse.Namespace(**PARAMETERS)
    logger = utils.get_logger(args.output_path)
    models = main_dense.load_models(args, logger)

    logger.info(DATASETS[i]["name"])
    logger.info(DATASETS[i]["model"])

    print(f'===Number of total entity in raw dataset=== {num_raw_data_entity[i]}')

    args = argparse.Namespace(**PARAMETERS)
    (
        biencoder_accuracy,
        recall_at,
        crossencoder_normalized_accuracy,
        overall_unormalized_accuracy,
        support,
        predictions,
        scores,
        biencoder_precision,
        biencoder_recall,
        biencoder_f1,
        crossencoder_precision,
        crossencoder_recall,
        crossencoder_f1,
        pred_entity_counter
    ) = main_dense.run(args, logger, *models)

    table.add_row(
        [
            DATASETS[i]["name"],
            DATASETS[i]["model"],
            ### biencoder
            round(biencoder_accuracy, 3),
            round(biencoder_precision, 3),
            round(biencoder_recall, 3),
            round(biencoder_f1, 3),
            ### crossencoder
            round(crossencoder_normalized_accuracy, 3),
            round(crossencoder_precision, 3),
            round(crossencoder_recall, 3),
            round(crossencoder_f1, 3),
            support,
            pred_entity_counter/num_raw_data_entity[i]
        ]
    )

logger.info("\n{}".format(table))
