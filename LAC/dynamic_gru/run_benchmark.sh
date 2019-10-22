export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=../../:$PYTHONPATH

python train.py \
    --train_data ./data/data/train.tsv \
    --test_data ./data/data/test.tsv \
    --model_save_dir ./padding_models \
    --validation_steps 2000 \
    --save_steps 10000 \
    --print_steps 200 \
    --batch_size 400 \
    --epoch 10 \
    --traindata_shuffle_buffer 20000 \
    --word_emb_dim 128 \
    --grnn_hidden_dim 128 \
    --bigru_num 2 \
    --base_learning_rate 1e-3 \
    --emb_learning_rate 2 \
    --crf_learning_rate 0.2 \
    --word_dict_path ./conf/word.dic \
    --label_dict_path ./conf/tag.dic \
    --word_rep_dict_path ./conf/q2b.dic \
    --enable_ce false \
    --use_cuda true \
    --cpu_num 1