import json
from tqdm import tqdm

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")

### convert mintaka data into the format of the ELQ model
def prepare_mintaka(input_filepath, output_filepath):
    id2title = json.load(open('/workspace/BLINK/models/id2title.json')) ### dictionary of id: title
    title_list = list(id2title.values()) ### title
    id2text = json.load(open('/workspace/BLINK/models/id2text.json')) ### dictionary of id: description
    text_list = list(id2text.values()) ### description

    data = [json.loads(line) for line in open(input_filepath, 'r', encoding='utf-8')] 
    entity_in_kb = 0
    entity_not_in_kb = 0
    all_entity_count = 0
    sentence_count = 0
    with open(output_filepath, "w+") as fout:
        for d in tqdm(data):
            is_in_kb = False
            tokenized_mention_idxs = []
            label_id = []
            description = []
            entity_spans = []
            wikidata_id = []
            entity = []
            for i in range(len(d['entity_mentions'])):
                all_entity_count += 1
                ### check whether entity is in ELQ KB, if not then pass
                if d['entity_mentions'][i] in title_list:
                    try:
                        ### tokenize text
                        tokenized_text_ids = tokenizer(d['question'], add_special_tokens=False, return_attention_mask = False, return_token_type_ids=False, return_offsets_mapping=True)
                        offset_mapping_start = [s for s,_ in tokenized_text_ids['offset_mapping']]
                        offset_mapping_end = [e for _,e in tokenized_text_ids['offset_mapping']]
                        tokenized_mention_idx = [offset_mapping_start.index(d['entity_spans'][i][0]), offset_mapping_end.index(d['entity_spans'][i][1])+1]
                        tokenized_mention_idxs.append(tokenized_mention_idx)
                        ### get label_id from id2title
                        label_id.append(title_list.index(d['entity_mentions'][i]))
                        ### get wikipedia description
                        description.append(text_list[title_list.index(d['entity_mentions'][i])])
                        ###
                        entity_spans.append(d['entity_spans'][i])
                        wikidata_id.append(d['entity_qids'][i])
                        entity.append(d['entity_mentions'][i])
                        is_in_kb = True
                        entity_in_kb += 1
                    except: 
                        entity_not_in_kb += 1
                else:
                    entity_not_in_kb += 1
            if is_in_kb == True:
                datapoint = {
                    'id': str(sentence_count),
                    'text' : d['question'],
                    'mentions' : entity_spans,
                    'tokenized_text_ids' : tokenized_text_ids['input_ids'],
                    'tokenized_mention_idxs' : tokenized_mention_idxs,
                    'label_id' : label_id,
                    'wikidata_id' : wikidata_id,
                    'entity' : entity,
                    'label' : description
                }
                ### dump result to json file
                json.dump(datapoint, fout)
                fout.write("\n")
            sentence_count += 1

    print(f'=== Number of all entity in data === {all_entity_count}')
    print(f'=== Number of entity in KB === {entity_in_kb}')
    print(f'=== Number of entity not in KB === {entity_not_in_kb}')
    print(f'=== Number of sentence in data === {sentence_count}')

### mintaka_train_data_el.json 
mintaka_train = prepare_mintaka("/workspace/datasets/mintaka_train_data_el.json", "/workspace/datasets/elq/mintaka/tokenized/train.jsonl")

### mintaka_dev_data_el.json 
mintaka_dev = prepare_mintaka("/workspace/datasets/mintaka_dev_data_el.json", "/workspace/datasets/elq/mintaka/tokenized/dev.jsonl")

### mintaka_test_data_el.json 
mintaka_test = prepare_mintaka("/workspace/datasets/mintaka_test_data_el.json", "/workspace/datasets/elq/mintaka/tokenized/test.jsonl")
