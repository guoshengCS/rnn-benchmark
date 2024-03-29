#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
import numpy as np
from paddle.fluid import ParamAttr
from paddle.fluid.contrib.layers import basic_lstm


def lm_model(hidden_size,
             vocab_size,
             batch_size,
             num_layers=2,
             num_steps=20,
             init_scale=0.1,
             dropout=None,
             rnn_model='static',
             use_dataloader=False):
    if rnn_model=='lod':
        x = fluid.data(name="x", shape=[None, 1], dtype='int64', lod_level=1)
        y = fluid.data(name="y", shape=[None, 1], dtype='int64', lod_level=1)

        if use_dataloader:
            dataloader = fluid.io.DataLoader.from_generator(
                feed_list=[x, y],
                capacity=16,
                iterable=False,
                use_double_buffer=True)

        init_hidden = fluid.data(name="init_hidden",
                                 shape=[None, num_layers, hidden_size],
                                 dtype='float32')
        init_cell = fluid.data(name="init_cell",
                               shape=[None, num_layers, hidden_size],
                               dtype='float32')

        init_cell.persistable = True
        init_hidden.persistable = True

        x_emb = layers.embedding(
            input=x,
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))

        if dropout != None and dropout > 0.0:
            x_emb = layers.dropout(
                x_emb,
                dropout_prob=dropout,
                dropout_implementation='upscale_in_train')

        lstm_input = x_emb
        last_hidden_array = []
        last_cell_array = []
        for i in range(num_layers):
            lstm_input = fluid.layers.fc(input=lstm_input,
                                         size=hidden_size * 4,
                                         bias_attr=False)
            hidden, cell = fluid.layers.dynamic_lstm(
                input=lstm_input,
                size=hidden_size * 4,
                h_0=init_hidden[:, i, :],
                c_0=init_cell[:, i, :],
                use_peepholes=False,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-init_scale, high=init_scale)))
            last_hidden = layers.sequence_pool(hidden, pool_type='last')
            last_cell = layers.sequence_pool(cell, pool_type='last')
            last_hidden_array.append(last_hidden)
            last_cell_array.append(last_cell)

            lstm_input = hidden
            if dropout != None and dropout > 0.0:
                lstm_input = layers.dropout(
                    lstm_input,
                    dropout_prob=dropout,
                    dropout_implementation='upscale_in_train')


        last_hidden = layers.stack(last_hidden_array, 1)
        last_cell = layers.stack(last_cell_array, 1)

        softmax_weight = layers.create_parameter(
            [hidden_size, vocab_size],
            dtype="float32",
            name="softmax_weight",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale))
        softmax_bias = layers.create_parameter(
            [vocab_size],
            dtype="float32",
            name='softmax_bias',
            default_initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale))

        projection = layers.matmul(hidden, softmax_weight)
        projection = layers.elementwise_add(projection, softmax_bias, axis=-1)

        loss = layers.softmax_with_cross_entropy(
            logits=projection, label=y, soft_label=False)
        loss = layers.sequence_pool(loss, pool_type='sum')
        loss = layers.reduce_mean(loss)

        feeding_list = ['x', 'y', 'init_hidden', 'init_cell']
        if use_dataloader:
            return loss, last_hidden, last_cell, feeding_list, dataloader
        else:
            return loss, last_hidden, last_cell, feeding_list

    def seq2seq_api_rnn(input_embedding, len=3, init_hiddens=None, init_cells=None):
        class EncoderCell(layers.RNNCell):
            def __init__(self,
                         num_layers,
                         hidden_size,
                         dropout_prob=0.,
                         forget_bias=0.):
                self.num_layers = num_layers
                self.hidden_size = hidden_size
                self.dropout_prob = dropout_prob
                self.lstm_cells = []
                for i in range(num_layers):
                    self.lstm_cells.append(
                        layers.LSTMCell(
                            hidden_size,
                            forget_bias=forget_bias,
                            param_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.
                                UniformInitializer(low=-init_scale,
                                                   high=init_scale))))

            def call(self, step_input, states):
                new_states = []
                for i in range(self.num_layers):
                    out, new_state  = self.lstm_cells[i](step_input, states[i])
                    step_input = layers.dropout(
                        out,
                        self.dropout_prob,
                        dropout_implementation='upscale_in_train'
                    ) if self.dropout_prob > 0 else out
                    new_states.append(new_state)
                return step_input, new_states

        cell = EncoderCell(num_layers, hidden_size, dropout)
        output, new_states = layers.rnn(
            cell,
            inputs=input_embedding,
            initial_states=[[hidden, cell] for hidden, cell in zip([
                layers.reshape(init_hidden, shape=[-1, hidden_size])
                for init_hidden in layers.split(
                    init_hiddens, num_or_sections=num_layers, dim=0)
            ], [
                layers.reshape(init_cell, shape=[-1, hidden_size])
                for init_cell in layers.split(
                    init_cells, num_or_sections=num_layers, dim=0)
            ])],
            time_major=False)
        last_hidden = layers.stack([hidden for hidden, _ in new_states], 0)
        last_cell = layers.stack([cell for _, cell in new_states], 0)
        return output, last_hidden, last_cell

    def padding_rnn(input_embedding, len=3, init_hidden=None, init_cell=None):
        weight_1_arr = []
        weight_2_arr = []
        bias_arr = []
        hidden_array = []
        cell_array = []
        mask_array = []
        for i in range(num_layers):
            weight_1 = layers.create_parameter(
                [hidden_size * 2, hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale))
            weight_1_arr.append(weight_1)
            bias_1 = layers.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            bias_arr.append(bias_1)

            pre_hidden = layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = layers.reshape(pre_hidden, shape=[-1, hidden_size])
            pre_cell = layers.reshape(pre_cell, shape=[-1, hidden_size])
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        input_embedding = layers.transpose(input_embedding, perm=[1, 0, 2])
        rnn = PaddingRNN()

        with rnn.step():
            input = rnn.step_input(input_embedding)
            for k in range(num_layers):
                pre_hidden = rnn.memory(init=hidden_array[k])
                pre_cell = rnn.memory(init=cell_array[k])
                weight_1 = weight_1_arr[k]
                bias = bias_arr[k]

                nn = layers.concat([input, pre_hidden], 1)
                gate_input = layers.matmul(x=nn, y=weight_1)

                gate_input = layers.elementwise_add(gate_input, bias)
                i = layers.slice(
                    gate_input, axes=[1], starts=[0], ends=[hidden_size])
                j = layers.slice(
                    gate_input,
                    axes=[1],
                    starts=[hidden_size],
                    ends=[hidden_size * 2])
                f = layers.slice(
                    gate_input,
                    axes=[1],
                    starts=[hidden_size * 2],
                    ends=[hidden_size * 3])
                o = layers.slice(
                    gate_input,
                    axes=[1],
                    starts=[hidden_size * 3],
                    ends=[hidden_size * 4])

                c = pre_cell * layers.sigmoid(f) + layers.sigmoid(
                    i) * layers.tanh(j)
                m = layers.tanh(c) * layers.sigmoid(o)

                rnn.update_memory(pre_hidden, m)
                rnn.update_memory(pre_cell, c)

                rnn.step_output(m)
                rnn.step_output(c)

                input = m

                if dropout != None and dropout > 0.0:
                    input = layers.dropout(
                        input,
                        dropout_prob=dropout,
                        dropout_implementation='upscale_in_train')

            rnn.step_output(input)
        rnnout = rnn()

        last_hidden_array = []
        last_cell_array = []
        real_res = rnnout[-1]
        for i in range(num_layers):
            m = rnnout[i * 2]
            c = rnnout[i * 2 + 1]
            m.stop_gradient = True
            c.stop_gradient = True
            last_h = layers.slice(
                m, axes=[0], starts=[num_steps - 1], ends=[num_steps])
            last_hidden_array.append(last_h)
            last_c = layers.slice(
                c, axes=[0], starts=[num_steps - 1], ends=[num_steps])
            last_cell_array.append(last_c)
        real_res = layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = layers.concat(last_hidden_array, 0)
        last_cell = layers.concat(last_cell_array, 0)

        return real_res, last_hidden, last_cell

    def encoder_static(input_embedding, len=3, init_hidden=None,
                       init_cell=None):

        weight_1_arr = []
        weight_2_arr = []
        bias_arr = []
        hidden_array = []
        cell_array = []
        mask_array = []
        for i in range(num_layers):
            weight_1 = layers.create_parameter(
                [hidden_size * 2, hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale))
            weight_1_arr.append(weight_1)
            bias_1 = layers.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            bias_arr.append(bias_1)

            pre_hidden = layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = layers.reshape(
                pre_hidden, shape=[-1, hidden_size], inplace=True)
            pre_cell = layers.reshape(
                pre_cell, shape=[-1, hidden_size], inplace=True)
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        res = []
        sliced_inputs = layers.split(
            input_embedding, num_or_sections=len, dim=1)

        for index in range(len):
            input = sliced_inputs[index]
            input = layers.reshape(input, shape=[-1, hidden_size], inplace=True)
            for k in range(num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = weight_1_arr[k]
                bias = bias_arr[k]

                nn = layers.concat([input, pre_hidden], 1)
                gate_input = layers.matmul(x=nn, y=weight_1)

                gate_input = layers.elementwise_add(gate_input, bias)
                i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)

                try:
                    from paddle.fluid.contrib.layers import fused_elemwise_activation
                    # fluid.contrib.layers.fused_elemwise_activation can do a fused
                    # operation, like:
                    # 1) x + sigmoid(y); x + tanh(y)
                    # 2) tanh(x + y)
                    # Now the unary operation supported in this fused op is limit, and
                    # we will extent this operation to support more unary operations and
                    # do this kind of fusion automitically in future version of paddle.fluid.
                    # layers.sigmoid(i) * layers.tanh(j)
                    tmp0 = fused_elemwise_activation(
                        x=layers.tanh(j),
                        y=i,
                        functor_list=['elementwise_mul', 'sigmoid'],
                        save_intermediate_out=False)
                    # pre_cell * layers.sigmoid(f)
                    tmp1 = fused_elemwise_activation(
                        x=pre_cell,
                        y=f,
                        functor_list=['elementwise_mul', 'sigmoid'],
                        save_intermediate_out=False)
                    c = tmp0 + tmp1
                    # layers.tanh(c) * layers.sigmoid(o)
                    m = fused_elemwise_activation(
                        x=layers.tanh(c),
                        y=o,
                        functor_list=['elementwise_mul', 'sigmoid'],
                        save_intermediate_out=False)
                except ImportError:
                    c = pre_cell * layers.sigmoid(f) + layers.sigmoid(
                        i) * layers.tanh(j)
                    m = layers.tanh(c) * layers.sigmoid(o)

                hidden_array[k] = m
                cell_array[k] = c
                input = m

                if dropout != None and dropout > 0.0:
                    input = layers.dropout(
                        input,
                        dropout_prob=dropout,
                        dropout_implementation='upscale_in_train')

            res.append(input)

        last_hidden = layers.concat(hidden_array, 1)
        last_hidden = layers.reshape(
            last_hidden, shape=[-1, num_layers, hidden_size], inplace=True)
        last_hidden = layers.transpose(x=last_hidden, perm=[1, 0, 2])

        last_cell = layers.concat(cell_array, 1)
        last_cell = layers.reshape(
            last_cell, shape=[-1, num_layers, hidden_size])
        last_cell = layers.transpose(x=last_cell, perm=[1, 0, 2])

        real_res = layers.concat(res, 0)
        real_res = layers.reshape(
            real_res, shape=[len, -1, hidden_size], inplace=True)
        real_res = layers.transpose(x=real_res, perm=[1, 0, 2])

        return real_res, last_hidden, last_cell

    batch_size_each = batch_size // fluid.core.get_cuda_device_count()
    x = fluid.data(
        # name="x", shape=[batch_size_each, num_steps, 1], dtype='int64')
        name="x",
        shape=[None, num_steps, 1],
        dtype='int64')
    y = fluid.data(
        # name="y", shape=[batch_size_each * num_steps, 1], dtype='int64')
        name="y",
        shape=[None, 1],
        dtype='int64')

    if use_dataloader:
        dataloader = fluid.io.DataLoader.from_generator(
            feed_list=[x, y],
            capacity=16,
            iterable=False,
            use_double_buffer=True)

    init_hidden = fluid.data(
        name="init_hidden",
        # shape=[num_layers, batch_size_each, hidden_size],
        shape=[num_layers, None, hidden_size],
        dtype='float32')
    init_cell = fluid.data(
        name="init_cell",
        # shape=[num_layers, batch_size_each, hidden_size],
        shape=[num_layers, None, hidden_size],
        dtype='float32')

    init_cell.persistable = True
    init_hidden.persistable = True

    init_hidden_reshape = layers.reshape(
        init_hidden, shape=[num_layers, -1, hidden_size])
    init_cell_reshape = layers.reshape(
        init_cell, shape=[num_layers, -1, hidden_size])

    x_emb = layers.embedding(
        input=x,
        size=[vocab_size, hidden_size],
        dtype='float32',
        is_sparse=False,
        param_attr=fluid.ParamAttr(
            name='embedding_para',
            initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale)))

    x_emb = layers.reshape(
        x_emb, shape=[-1, num_steps, hidden_size], inplace=True)
    if dropout != None and dropout > 0.0:
        x_emb = layers.dropout(
            x_emb,
            dropout_prob=dropout,
            dropout_implementation='upscale_in_train')

    if rnn_model == "padding":
        rnn_out, last_hidden, last_cell = padding_rnn(
            x_emb,
            len=num_steps,
            init_hidden=init_hidden_reshape,
            init_cell=init_cell_reshape)
    elif rnn_model == "static":
        rnn_out, last_hidden, last_cell = encoder_static(
            x_emb,
            len=num_steps,
            init_hidden=init_hidden_reshape,
            init_cell=init_cell_reshape)
    elif rnn_model == "cudnn":
        x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
        rnn_out, last_hidden, last_cell = layers.lstm(
            x_emb,
            init_hidden_reshape,
            init_cell_reshape,
            num_steps,
            hidden_size,
            num_layers,
            is_bidirec=False,
            default_initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale))
        rnn_out = layers.transpose(rnn_out, perm=[1, 0, 2])
    elif rnn_model == "basic_lstm":
        rnn_out, last_hidden, last_cell = basic_lstm( x_emb, init_hidden, init_cell, hidden_size, \
                num_layers=num_layers, batch_first=True, dropout_prob=dropout, \
                param_attr = ParamAttr( initializer=fluid.initializer.UniformInitializer(low=-init_scale, high=init_scale) ), \
                bias_attr = ParamAttr( initializer = fluid.initializer.Constant(0.0) ), \
                forget_bias = 0.0)
    elif rnn_model == "seq2seq_api":
        rnn_out, last_hidden, last_cell = seq2seq_api_rnn(
            x_emb,
            len=num_steps,
            init_hiddens=init_hidden_reshape,
            init_cells=init_cell_reshape)
    else:
        print("type not support")
        return

    rnn_out = layers.reshape(
        rnn_out, shape=[-1, num_steps, hidden_size], inplace=True)

    softmax_weight = layers.create_parameter(
        [hidden_size, vocab_size],
        dtype="float32",
        name="softmax_weight",
        default_initializer=fluid.initializer.UniformInitializer(
            low=-init_scale, high=init_scale))
    softmax_bias = layers.create_parameter(
        [vocab_size],
        dtype="float32",
        name='softmax_bias',
        default_initializer=fluid.initializer.UniformInitializer(
            low=-init_scale, high=init_scale))

    projection = layers.matmul(rnn_out, softmax_weight)
    projection = layers.elementwise_add(projection, softmax_bias)
    projection = layers.reshape(
        projection, shape=[-1, vocab_size], inplace=True)

    loss = layers.softmax_with_cross_entropy(
        logits=projection, label=y, soft_label=False)

    loss = layers.reshape(loss, shape=[-1, num_steps], inplace=True)
    loss = layers.reduce_mean(loss, dim=[0])
    loss = layers.reduce_sum(loss)

    loss.persistable = True
    last_cell.persistable = True
    last_hidden.persistable = True

    # This will feed last_hidden, last_cell to init_hidden, init_cell, which
    # can be used directly in next batch. This can avoid the fetching of
    # last_hidden and last_cell and feeding of init_hidden and init_cell in
    # each training step.
    layers.assign(input=last_cell, output=init_cell)
    layers.assign(input=last_hidden, output=init_hidden)

    feeding_list = ['x', 'y', 'init_hidden', 'init_cell']
    if use_dataloader:
        return loss, last_hidden, last_cell, feeding_list, dataloader
    else:
        return loss, last_hidden, last_cell, feeding_list
