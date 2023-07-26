import json
from tqdm import tqdm

### load all the 5903527 entities
def load_entities():
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_id2wikipedia_id = {}
    local_idx = 0
    with open('/workspace/BLINK/models/entity.jsonl', "r") as fin:
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

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return title2id, id2title, id2text, wikipedia_id2local_id, local_id2wikipedia_id

title2id, id2title, id2text, wikipedia_id2local_id, local_id2wikipedia_id = load_entities()

### convert mintaka data into the format of the BLINK model
def prepare_mintaka(input_filepath, output_filepath):
    data = [json.loads(line) for line in open(input_filepath, 'r', encoding='utf-8')]
    all_entity_count = 0
    entity_not_in_KB = 0
    entity_in_KB = 0
    sentence_count = 0
    with open(output_filepath, "w+") as fout:
        for d in tqdm(data):
            for span, entity in zip(d['entity_spans'], d['entity_mentions']):
                all_entity_count += 1
                try:
                    local_id = title2id[entity]
                    wikipedia_id = local_id2wikipedia_id[local_id]
                    datapoint = {
                        "context_left": d['question'][:span[0]-1],
                        "mention": entity,
                        "context_right": d['question'][span[1]+1:],
                        "query_id": str(entity_in_KB),
                        "label_id": wikipedia_id,
                        "label": id2text[local_id],
                        "label_title": id2title[local_id]
                    }
                    json.dump(datapoint, fout)
                    fout.write("\n")
                    entity_in_KB += 1
                except: 
                    entity_not_in_KB += 1
            sentence_count += 1
    
    print(f'Number of total entity in data: {all_entity_count}')
    print(f'Number of entity in KB: {entity_in_KB}')
    print(f'Number of entity not in KB: {entity_not_in_KB}')
    print(f'Number of sentence in data: {sentence_count}')

### mintaka_train_data_el.json 
mintaka_train = prepare_mintaka("/workspace/datasets/mintaka_train_data_el.json", "/workspace/datasets/data4blinkv2/mintaka/train.jsonl")

# ### mintaka_dev_data_el.json 
mintaka_dev = prepare_mintaka("/workspace/datasets/mintaka_dev_data_el.json", "/workspace/datasets/data4blinkv2/mintaka/valid.jsonl")

# ### mintaka_test_data_el.json 
mintaka_test = prepare_mintaka("/workspace/datasets/mintaka_test_data_el.json", "/workspace/datasets/data4blinkv2/mintaka/test.jsonl")