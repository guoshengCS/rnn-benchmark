export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=../../:$PYTHONPATH

python train.py \
    --data_path data/simple-examples/data/ \
    --model_type small \
    --use_gpu True \
    --rnn_model lod \
    --parallel True