import json
from tqdm import tqdm

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")


def preprare_shadow(input_filepath, output_filepath):
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
            all_entity_count += 1
            ### check whether entity is in ELQ KB, if not then pass
            if d['wiki_title'] in title_list:
                ### tokenize text
                tokenized_mention_idxs = []
                tokenized_text_ids = tokenizer(d['example'], add_special_tokens=False, return_attention_mask = False, return_token_type_ids=False, return_offsets_mapping=True)
                offset_mapping_start = [s for s,e in tokenized_text_ids['offset_mapping']]
                offset_mapping_end = [e for s,e in tokenized_text_ids['offset_mapping']]
                start = d['span'][0]
                end = d['span'][0]+d['span'][1]
                try:
                    tokenized_mention_idx = [offset_mapping_start.index(start), offset_mapping_end.index(end)+1]
                    tokenized_mention_idxs.append(tokenized_mention_idx)
                    ### get label_id from id2title
                    label_id = title_list.index(d['wiki_title'])

                    datapoint = {
                        'id': str(sentence_count),
                        'text' : d['example'],
                        'mentions' : [[start, end]],
                        'tokenized_text_ids' : tokenized_text_ids['input_ids'],
                        'tokenized_mention_idxs' : tokenized_mention_idxs,
                        'label_id' : [label_id],
                        'wikidata_id' : [str(d['wiki_id'])],
                        'entity' : [d['wiki_title']],
                        'label' : [text_list[label_id]]
                        }

                    ### dump result to json file
                    json.dump(datapoint, fout)
                    fout.write("\n")
                    entity_in_kb += 1
                except:
                    entity_not_in_kb += 1
            else:
                entity_not_in_kb += 1
        sentence_count += 1
    
    print(f'=== Number of all entity in data === {all_entity_count}')
    print(f'=== Number of entity in KB === {entity_in_kb}')
    print(f'=== Number of entity not in KB === {entity_not_in_kb}')
    print(f'=== Number of sentence in data === {sentence_count}')

## shadowlink_top.json 
shadow_top = preprare_shadow('/workspace/datasets/shadowlink_top.json', '/workspace/datasets/elq/shadowlink/tokenized/top.jsonl')

## shadowlink_tail.json 
shadow_tail = preprare_shadow('/workspace/datasets/shadowlink_tail.json', '/workspace/datasets/elq/shadowlink/tokenized/tail.jsonl')

### shadowlink_shadow.json 
shadow_shadow = preprare_shadow('/workspace/datasets/shadowlink_shadow.json', '/workspace/datasets/elq/shadowlink/tokenized/shadow.jsonl')