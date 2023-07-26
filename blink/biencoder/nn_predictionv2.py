# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):
    ### get model + data
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    ### get entity 
    wikipedia_id2local_id = {}
    local_id2wikipedia_id = {}
    local_idx = 0
    with open("models/entity.jsonl", "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx
                local_id2wikipedia_id[local_idx] = wikipedia_id
            local_idx += 1
    
    nn_context = []
    nn_candidates = []
    nn_labels = []
    num_pred_correct = 0
    num_all_entity = 0
    candidate_pool = candidate_pool.to(device)
    
    oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, _, label_wiki_ids = batch
    
        with torch.no_grad():
            scores = reranker.score_candidate(
                context_input, 
                None, 
                cand_encs=cand_encode_list.to(device)
            )
            _, indicies = scores.topk(top_k)

        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i] ### indices = local_id

            ### convert predicted topk entity from wikipedia_id to local_id
            label_local_id = wikipedia_id2local_id[label_wiki_ids[i].item()]

            pointer = -1
            for t in range(top_k):
                if inds[t].item() == label_local_id:
                    num_pred_correct += 1
                    pointer = t
                    break

            if pointer == -1:
                continue

            if not save_predictions:
                continue

            # add examples in new_data
            cur_candidates = candidate_pool[inds]
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)

        num_all_entity += context_input.size(0)

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
    }

    logger.info("Recall@100: %d " % num_pred_correct/num_all_entity)
    
    return nn_data

