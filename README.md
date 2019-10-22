
PaddlePaddle中RNN具有多种实现方式，期望给出不同实现方式在不同场景下的性能对比。

暂时选取了LAC和LM两个任务场景，给出了各自以下方式的实现（每种实现方式都有单独对应的目录，每个目录内的内容可通过`run_benchmark.sh`运行）：

```text
.
├── LAC                        # LAC 各种实现方式
    ├── dynamic_gru            # LAC dynamic_gru
    └── seq2seq_api_gru        # LAC seq2seq_api_gru
├── LM                         # LM 各种实现方式
    ├── cudnn_lstm             # LM cudnn_lstm
    ├── dynamic_lstm           # LM dynamic_lstm
    └── seq2seq_api_lstm       # LM seq2seq_api_lstm
└── models                     # https://github.com/PaddlePaddle/models/tree/release/1.6/PaddleNLP/models
```

注：
 - LAC训练数据见 http://wiki.baidu.com/pages/viewpage.action?pageId=919502481
 - 待cudnn_gru合入Paddle后LAC加入相应的测试
 - LM目前只支持单卡