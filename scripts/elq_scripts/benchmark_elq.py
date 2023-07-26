# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import prettytable
import json
import elq.main_dense as main_dense
import elq.candidate_ranking.utils as utils

DATASETS = [
    {   "name": "aida_test",
        "model_name": "ft_aida",
        "filename": "/workspace/datasets/elq/aida/tokenized/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/elq_ft_models/aida_model/epoch_5/pytorch_model.bin",
        "biencoder_config": "/workspace/BLINK/elq_ft_models/aida_model/training_params.txt",
        "max_context_length": 512,
        "threshold": -2.9 
    },
    {   "name": "mintaka_test",
        "model_name": "ft_mintaka",
        "filename": "/workspace/datasets/elq/mintaka/tokenized/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/elq_ft_models/mintaka_model/epoch_23/pytorch_model.bin",
        "biencoder_config": "/workspace/BLINK/elq_ft_models/mintaka_model/training_params.txt",
        "max_context_length": 32,
        "threshold": -2.9 
    },
    {   "name": "webqsp_test",
        "model_name": "ft_webqsp",
        "filename": "/workspace/datasets/elq/webqsp/tokenized/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/elq_ft_models/webqsp_model/epoch_51/pytorch_model.bin",
        "biencoder_config": "/workspace/BLINK/elq_ft_models/webqsp_model/training_params.txt",
        "max_context_length": 32,
        "threshold": -1.5
    },
    {   "name": "graphq_test",
        "model_name": "ft_graphq",
        "filename": "/workspace/datasets/elq/graphq/tokenized/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/elq_ft_models/graphq_model/epoch_90/pytorch_model.bin",
        "biencoder_config": "/workspace/BLINK/elq_ft_models/graphq_model/training_params.txt",
        "max_context_length": 32,
        "threshold": -0.9 
    },
    {   "name": "shadow_top",
        "model_name": "wiki",
        "filename": "/workspace/datasets/data4elq/shadowlink/top.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 32,
        "threshold": -2.9 
    },
    {   "name": "shadow_tail",
        "model_name": "wiki",
        "filename": "/workspace/datasets/data4elq/shadowlink/tail.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 32,
        "threshold": -2.9 
    },
    {   "name": "shadow_shadow",
        "model_name": "wiki",
        "filename": "/workspace/datasets/data4elq/shadowlink/shadow.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 32,
        "threshold": -2.9 
    },
    {   "name": "ace2004",
        "model_name": "wiki",
        "filename": "/workspace/datasets/elq/ace2004/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 512,
        "threshold": -2.9 
    },
    {   "name": "aquaint",
        "model_name": "wiki",
        "filename": "/workspace/datasets/elq/aquaint/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 512,
        "threshold": -2.9 
    },
    {   "name": "clueweb",
        "model_name": "wiki",
        "filename": "/workspace/datasets/elq/clueweb/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 512,
        "threshold": -2.9 
    },
    {   "name": "msnbc",
        "model_name": "wiki",
        "filename": "/workspace/datasets/elq/msnbc/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 512,
        "threshold": -2.9 
    },
    {   "name": "wikipedia",
        "model_name": "wiki",
        "filename": "/workspace/datasets/elq/wiki/test.jsonl",
        "save_preds_dir": None,
        "biencoder_model": "/workspace/BLINK/models/elq_wiki_large.bin",
        "biencoder_config": "/workspace/BLINK/models/elq_large_params.txt",
        "max_context_length": 512,
        "threshold": -2.9 
    },
]

raw_data = [
    "/workspace/datasets/aida_test.json", ### aida
    "/workspace/datasets/mintaka_test_data_el.json", ### mintaka
    "/workspace/datasets/webqsp_test_data_el.json", ### webqsp
    "/workspace/datasets/webqsp_test_data_el.json", ### webqsp
    "/workspace/datasets/graphq_test_data_el.json", ### graphq
    "/workspace/datasets/graphq_test_data_el.json", ### graphq
    "/workspace/datasets/shadowlink_top.json", ### shadow_top
    "/workspace/datasets/shadowlink_tail.json", ### shadow_tail
    "/workspace/datasets/shadowlink_shadow.json", ### shadow_shadow
    "/workspace/datasets/ace2004_parsed.json",
    "/workspace/datasets/aquaint_parsed.json",
    "/workspace/datasets/clueweb_parsed.json",
    "/workspace/datasets/msnbc_parsed.json",
    "/workspace/datasets/wikipedia_parsed.json",
]

### get number of total entity in dataset
### spans, entity_mentions, entity, entity_name
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
        if 'mentions' in d.keys():
            entity_count += len(d['mentions'])
    num_raw_data_entity.append(entity_count)

PARAMETERS = {
    "eval_batch_size": 64,
    "output_path": "elq_output",
    ###
    "debug_biencoder": False,
    "get_predictions": False,
    "interactive": False,
    ###
    "test_entities": "models/entity.jsonl",
    "mention_threshold": None,
    "num_cand_mentions": 50,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    ### biencoder
    "cand_token_ids_path": "models/entity_token_ids_128.t7",
    "entity_catalogue": "models/entity.jsonl",
    "entity_encoding": "models/all_entities_large.t7",
    "faiss_index": "hnsw",
    "index_path": "models/faiss_hnsw_index.pkl",
    ###
    "use_cuda": False,
    "no_logger": False
}

table = prettytable.PrettyTable(
    [
        "DATASET",
        "MODEL",
        "Context length",
        "Threshold", 
        "In KB entity precision",
        "In KB entity recall",
        "In KB entity F1-score",
        "All entity precision",
        "All entity recall",
        "All entity f1-score",
        "Ratio of in KB entity/All entity"
    ]
)

for i in range(len(DATASETS)):
    PARAMETERS["test_mentions"] = DATASETS[i]["filename"]
    PARAMETERS["save_preds_dir"] = DATASETS[i]["save_preds_dir"]
    PARAMETERS["biencoder_model"] = DATASETS[i]["biencoder_model"]
    PARAMETERS["biencoder_config"] = DATASETS[i]["biencoder_config"]
    PARAMETERS["max_context_length"] = DATASETS[i]["max_context_length"]
    PARAMETERS["threshold"] = DATASETS[i]["threshold"]

    args = argparse.Namespace(**PARAMETERS)
    logger = utils.get_logger(args.output_path)
    models = main_dense.load_models(args, logger)

    logger.info(DATASETS[i]["name"])
    (   _, num_test_set_entity, tp, fp, fn, precision, recall, f1
    ) = main_dense.run(args, logger, *models)

    print(f'===Number of total entity in raw dataset=== {num_raw_data_entity[i]}')
    print(f'===Number of predicted entity=== {num_test_set_entity}')
    all_precision = tp / (tp + (fp + num_raw_data_entity[i]-num_test_set_entity))
    all_recall = tp / (tp + (fn + num_raw_data_entity[i]-num_test_set_entity))
    all_f1 = (2*all_precision*all_recall)/(all_precision+all_recall)

    table.add_row(
        [
            DATASETS[i]["name"],
            DATASETS[i]["model_name"],
            DATASETS[i]["max_context_length"],
            DATASETS[i]["threshold"], 
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            ### raw data precision-recall-f1score
            round(all_precision, 4),
            round(all_recall, 4),
            round(all_f1, 4),
            num_test_set_entity/num_raw_data_entity[i]
        ]
    )

logger.info("\n{}".format(table))
logger.info("\n All data = considered all entity even if they're not in KB. != test data contains only entity in KB")