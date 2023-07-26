import json
from tqdm import tqdm


from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")

def prepare_additional_data(input_filepath, output_filepath):
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
            mention_char_span = []
            label_id = []
            description = []
            entity = []

            ### tokenize text
            tokenized_text_ids = tokenizer(d['text'], add_special_tokens=False, return_attention_mask = False, return_token_type_ids=False, return_offsets_mapping=True)
            offset_mapping_start = [s for s,_ in tokenized_text_ids['offset_mapping']]
            offset_mapping_end = [e for _,e in tokenized_text_ids['offset_mapping']]

            ### check whether entity is in ELQ KB, if not then pass
            for mention in d['mentions']:
                all_entity_count += 1
                if mention['wiki_name'] in title_list:
                    try:
                        tokenized_mention_idx = [offset_mapping_start.index(mention['start']), offset_mapping_end.index(mention['start'] + mention['length'])+1]
                        tokenized_mention_idxs.append(tokenized_mention_idx)
                        ### get mention character span
                        mention_char_span.append([mention['start'], mention['start'] + mention['length']])
                        ### get label_id from id2title
                        label_id.append(title_list.index(mention['wiki_name']))
                        ### get wikipedia description
                        description.append(text_list[title_list.index(mention['wiki_name'])])
                        entity.append(mention['text'])
                        is_in_kb = True
                        entity_in_kb += 1
                    except: 
                        entity_not_in_kb += 1
                else:
                    entity_not_in_kb += 1
            if is_in_kb == True:
                datapoint = {
                    'id': d['doc_title'],
                    'text' : d['text'],
                    'mentions' : mention_char_span,
                    'tokenized_text_ids' : tokenized_text_ids['input_ids'],
                    'tokenized_mention_idxs' : tokenized_mention_idxs,
                    'label_id' : label_id,
                    'wikidata_id' : [str(id) for id in label_id],
                    'entity' : entity,
                    'label' : description
                }
                ### dump result to json file
                json.dump(datapoint, fout)
                fout.write("\n")
            else:
                pass
            sentence_count += 1
    
    print(f'=== Number of all entity in data === {all_entity_count}')
    print(f'=== Number of entity in KB === {entity_in_kb}')
    print(f'=== Number of entity not in KB === {entity_not_in_kb}')
    print(f'=== Number of sentence in data === {sentence_count}')

### ace2004
ace = prepare_additional_data("/workspace/datasets/ace2004_parsed.json", "/workspace/datasets/elq/ace2004/test.jsonl")

### aquaint
aquaint = prepare_additional_data("/workspace/datasets/aquaint_parsed.json", "/workspace/datasets/elq/aquaint/test.jsonl")

### clueweb
clueweb = prepare_additional_data("/workspace/datasets/clueweb_parsed.json", "/workspace/datasets/elq/clueweb/test.jsonl")

### msnbc
msnbc = prepare_additional_data("/workspace/datasets/msnbc_parsed.json", "/workspace/datasets/elq/msnbc/test.jsonl")

### wiki
wiki = prepare_additional_data("/workspace/datasets/wikipedia_parsed.json", "/workspace/datasets/elq/wiki/test.jsonl")