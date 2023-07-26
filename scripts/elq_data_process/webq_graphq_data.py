import json
from tqdm import tqdm

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")

def prepare_q(input_filepath, output_filepath):
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
            ### check whether entity is in ELQ KB, if not then pass
            for i in range(len(d['entity'])):
                all_entity_count += 1
                if d['entity'][i] in title_list:
                    try:
                        ### tokenize text
                        tokenized_text_ids = tokenizer(d['text'], add_special_tokens=False, return_attention_mask = False, return_token_type_ids=False, return_offsets_mapping=True)
                        offset_mapping_start = [s for s,_ in tokenized_text_ids['offset_mapping']]
                        offset_mapping_end = [e for _,e in tokenized_text_ids['offset_mapping']]
                        tokenized_mention_idx = [offset_mapping_start.index(d['mentions'][i][0]), offset_mapping_end.index(d['mentions'][i][1])+1]
                        tokenized_mention_idxs.append(tokenized_mention_idx)
                        ### get label_id from id2title
                        label_id.append(title_list.index(d['entity'][i]))
                        ### get wikipedia description
                        description.append(text_list[title_list.index(d['entity'][i])])
                        is_in_kb = True
                        entity_in_kb += 1
                    except: 
                        entity_not_in_kb += 1
                else:
                    entity_not_in_kb += 1
            if is_in_kb == True:
                datapoint = {
                    'id': str(d['id']),
                    'text' : d['text'],
                    'mentions' : d['mentions'],
                    'tokenized_text_ids' : tokenized_text_ids['input_ids'],
                    'tokenized_mention_idxs' : tokenized_mention_idxs,
                    'label_id' : d['label_id'],
                    'wikidata_id' : d['wikidata_id'],
                    'entity' : d['entity'],
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


### webqsp_training_data_el
webqsp_train = prepare_q("/workspace/datasets/webqsp_training_data_el.json", "/workspace/datasets/elq/webqsp/tokenized/train.jsonl")

### webqsp_dev_data_el
webqsp_dev = prepare_q("/workspace/datasets/webqsp_dev_data_el.json", "/workspace/datasets/elq/webqsp/tokenized/dev.jsonl")

### webqsp_test_data_el
webqsp_test = prepare_q("/workspace/datasets/webqsp_test_data_el.json", "/workspace/datasets/elq/webqsp/tokenized/test.jsonl")

### graphq_train_data_el
graphq_train = prepare_q("/workspace/datasets/graphq_train_data_el.json", "/workspace/datasets/elq/graphq/tokenized/train.jsonl")

### graphq_dev_data_el
graphq_dev = prepare_q("/workspace/datasets/graphq_dev_data_el.json", "/workspace/datasets/elq/graphq/tokenized/dev.jsonl")

### graphq_test_data_el
graphq_test = prepare_q("/workspace/datasets/graphq_test_data_el.json", "/workspace/datasets/elq/graphq/tokenized/test.jsonl")