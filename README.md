
PaddlePaddle中RNN具有多种实现方式，期望给出不同实现方式在不同场景下的性能对比。

暂时选取了LAC和LM两个任务场景，需给出各自以下方式的实现（每种实现方式都有单独对应的目录，保证每个的目录内可通过`run_benchmark.sh`独立运行）：

```text
.
├── LAC                        # LAC 各种实现方式
    ├── cudnn_lstm             # LAC cudnn_lstm
    ├── dynamic_lstm           # LAC dynamic_lstm
    └── seq2seq_api_lstm       # LAC seq2seq_api_lstm
├── LM                         # LM 各种实现方式
    ├── cudnn_lstm             # LM cudnn_lstm
    ├── dynamic_lstm           # LM dynamic_lstm
    └── seq2seq_api_lstm       # LM seq2seq_api_lstm
└── models                     # https://github.com/PaddlePaddle/models/tree/release/1.6/PaddleNLP/models
```

已从models拷贝了基础版本，各实现方式可以基于基础版本修改，部分实现已完成，修改事项如下：

- LAC: 使用LSTM替换GRU，同时去掉L2DecayRegularizer
  - dynamic_lstm
    - nets.py dynamic_gru->dynamic_lstm
  - cudnn_lstm
    - nets.py _bigru_layer->lstm(cudnn)
  - seq2seq_api_lstm
    - nets.py pre_gru去掉

- LM
  - dynamic_lstm
    - 新增
  - cudnn_lstm
    - Done
  - seq2seq_api_lstm
    - Done 



