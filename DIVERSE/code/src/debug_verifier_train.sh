export CUDA_VISIBLE_DEVICES=0
export MAX_LENGTH=512
export BERT_MODEL=microsoft/deberta-v3-large
export OUTPUT_DIR=../../debug
export BATCH_SIZE=
export NUM_EPOCHS=16
export SAVE_STEPS=10000
export SEED=1

# python -B -m torch.distributed.launch --nproc_per_node=1 --master_port=23333 run_ner.py \
# --task_type NER \
# --train_data /home/v-yifeili/blob/dkibdm/aml_components/deberta_verifier_training_stepwise/data_new/train.txt \
# --test_data /home/v-yifeili/blob/dkibdm/aml_components/deberta_verifier_training_stepwise/data_new/dev.txt \
# --data_labels /home/v-yifeili/blob/dkibdm/aml_components/deberta_verifier_training_stepwise/data_new/labels.txt \
# --model_name_or_path /home/v-yifeili/blob/dkibdm/math_word_problems/GSM8K_models/deberta_verifier_prompting_1k_20samples_lr1e-5_seed1_bs8/checkpoint-620 \
# --output_dir /home/v-yifeili/blob/dkibdm/math_word_problems/debug/deberta_token_level_stepwise/10epochs_lr1e-5_seed1_bs16 \
# --max_seq_length 512 \
# --num_train_epochs 10 \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 400 \
# --save_strategy epoch \
# --evaluation_strategy epoch \
# --learning_rate 1e-5 \
# --lr_scheduler_type constant \
# --seed 1 \
# --do_eval \
# --logging_steps 10 \
# --overwrite_output_dir \
# --solution_correct_loss_weight 1.0 \
# --solution_incorrect_loss_weight 1.0 \
# --step_correct_loss_weight 0.0 \
# --step_incorrect_loss_weight 0.0

# for debugging
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=23456 run_ner.py \
--task_type NER \
--train_data /home/lyf/projects/aml-babel-components/datasets/gsm8k_debug/train.txt \
--test_data /home/lyf/projects/aml-babel-components/datasets/gsm8k_debug/test.txt \
--data_labels /home/lyf/projects/aml-babel-components/gsm8k/src/labels.txt \
--dataset_name GSM8K \
--model_name_or_path microsoft/deberta-v3-xsmall \
--output_dir /home/lyf/projects/math_word_problems/debug/deberta \
--max_seq_length 512 \
--sentences_attention_weights_file /home/lyf/projects/math_word_problems/code/stepwise_verifier_tokencls_kl/weights_deberta_all_sentences_saved_for_debug.txt \
--num_train_epochs 10 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--save_strategy epoch \
--evaluation_strategy epoch \
--learning_rate 1e-5 \
--lr_scheduler_type constant \
--seed 1 \
--do_train \
--do_eval \
--logging_steps 10 \
--overwrite_output_dir \
--lambda_val 1.0