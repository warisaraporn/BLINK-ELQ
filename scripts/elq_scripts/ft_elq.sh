### own graphq data
CUDA_VISIBLE_DEVICES=0 python elq/biencoder/train_biencoder.py \
    --data_path /workspace/datasets/elq/graphq/bert_large_tokenized/ \
    --output_path /workspace/BLINK/elq_ft_models/graphq_bert_large_tokenize \
    --path_to_model /workspace/BLINK/models/elq_wiki_large.bin \
    --bert_model bert-large-uncased \
    --max_seq_length 256 --max_context_length 20 --max_cand_length 128 \
    --lowercase --context_key None --title_key entity \
    --max_mention_length 10 --mention_aggregation_type all_avg --no_mention_bounds \
    --train_batch_size 128 --eval_batch_size 64 --max_grad_norm 1.0 --learning_rate 0.00001 --num_train_epochs 100 \
    --eval_interval 500 --warmup_proportion 0.1 --freeze_cand_enc \
    --cand_enc_path models/all_entities_large.t7 --cand_token_ids_path models/entity_token_ids_128.t7 \
    --get_losses --adversarial_training

### own webqsp data
CUDA_VISIBLE_DEVICES=0 python elq/biencoder/train_biencoder.py \
    --data_path /workspace/datasets/elq/webqsp/bert_large_tokenized/ \
    --output_path /workspace/BLINK/elq_ft_models/webqsp_bert_large_tokenize \
    --path_to_model /workspace/BLINK/models/elq_wiki_large.bin \
    --bert_model bert-large-uncased \
    --max_seq_length 256 --max_context_length 20 --max_cand_length 128 \
    --lowercase --context_key None --title_key entity \
    --max_mention_length 10 --mention_aggregation_type all_avg --no_mention_bounds \
    --train_batch_size 128 --eval_batch_size 64 --max_grad_norm 1.0 --learning_rate 0.00001 --num_train_epochs 100 \
    --eval_interval 500 --warmup_proportion 0.1 --freeze_cand_enc \
    --cand_enc_path models/all_entities_large.t7 --cand_token_ids_path models/entity_token_ids_128.t7 \
    --get_losses --adversarial_training

### mintaka
CUDA_VISIBLE_DEVICES=1 python elq/biencoder/train_biencoder.py \
    --data_path /workspace/datasets/data4elq/mintaka/tokenized/ \
    --output_path /workspace/BLINK/elq_ft_modelsv3/mintaka_model \
    --path_to_model /workspace/BLINK/models/elq_wiki_large.bin \
    --bert_model bert-large-uncased \
    --max_seq_length 256 --max_context_length 20 --max_cand_length 128 \
    --lowercase --context_key None --title_key entity \
    --max_mention_length 10 --mention_aggregation_type all_avg --no_mention_bounds \
    --train_batch_size 128 --eval_batch_size 64 --max_grad_norm 1.0 --learning_rate 0.00001 --num_train_epochs 100 \
    --eval_interval 500 --warmup_proportion 0.1 --freeze_cand_enc \
    --cand_enc_path models/all_entities_large.t7 --cand_token_ids_path models/entity_token_ids_128.t7 \
    --get_losses --adversarial_training

### aida
CUDA_VISIBLE_DEVICES=1 python elq/biencoder/train_biencoder.py \
    --data_path /workspace/datasets/data4elq/aida/tokenized/ \
    --output_path /workspace/BLINK/elq_ft_modelsv3/aida_model \
    --path_to_model /workspace/BLINK/models/elq_wiki_large.bin \
    --bert_model bert-large-uncased \
    --max_seq_length 256 --max_context_length 20 --max_cand_length 128 \
    --lowercase --context_key None --title_key entity \
    --max_mention_length 10 --mention_aggregation_type all_avg --no_mention_bounds \
    --train_batch_size 128 --eval_batch_size 64 --max_grad_norm 1.0 --learning_rate 0.00001 --num_train_epochs 100 \
    --eval_interval 500 --warmup_proportion 0.1 --freeze_cand_enc \
    --cand_enc_path models/all_entities_large.t7 --cand_token_ids_path models/entity_token_ids_128.t7 \
    --get_losses --adversarial_training
