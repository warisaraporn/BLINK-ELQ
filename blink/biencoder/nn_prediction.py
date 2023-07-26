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
from blink.biencoder.zeshel_utils import WORLDS, Stats


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
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    ### create entity mapping between Wikipedia ID and local ID
    wikipedia_id2local_id = {}
    local_id2wikipedia_id = {}
    local_idx = 0
    with open("./models/entity.jsonl", "r") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
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
    nn_worlds = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)
    
    oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        if is_zeshel:
            context_input, _, srcs, label_ids = batch
        else:
            context_input, _, label_ids = batch
            srcs = torch.tensor([0] * context_input.size(0), device=device)

        src = srcs[0].item()
        scores = reranker.score_candidate(
            context_input, 
            None, 
            cand_encs=cand_encode_list[src].to(device)
        )
        values, indicies = scores.topk(top_k)
        old_src = src
        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i] ### local_id

            if srcs[i] != old_src:
                src = srcs[i].item()
                # not the same domain, need to re-do
                new_scores = reranker.score_candidate(
                    context_input[[i]], 
                    None,
                    cand_encs=cand_encode_list[src].to(device)
                )
                _, inds = new_scores.topk(top_k)
                inds = inds[0]

            pointer = -1
            local_id = wikipedia_id2local_id[label_ids[i].item()] ### convert from wikipedia_id to local_id
            for j in range(top_k):
                if inds[j].item() == local_id:
                    pointer = j
                    break
            stats[src].add(pointer)

            if pointer == -1:
                continue

            if not save_predictions:
                continue

            # add examples in new_data
            # cur_candidates = candidate_pool[src][inds]
            cur_candidates = candidate_pool[srcs[i].item()].to(device)[inds]
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_worlds.append(src)

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
    }

    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data

