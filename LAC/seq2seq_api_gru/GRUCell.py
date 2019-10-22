import paddle.fluid as fluid
from paddle.fluid import layers
from paddle.fluid.layers import RNNCell
from paddle.fluid.dygraph import Layer

class BasicGRUUnit_(Layer):
    """
    ****
    BasicGRUUnit class, using basic operators to build GRU
    The algorithm can be described as the equations below.

        .. math::
            u_t & = actGate(W_ux xu_{t} + W_uh h_{t-1} + b_u)

            r_t & = actGate(W_rx xr_{t} + W_rh h_{t-1} + b_r)

            m_t & = actNode(W_cx xm_t + W_ch dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    Args:
        name_scope(string) : The name scope used to identify parameters and biases
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of GRU unit.
            If it is set to None or one attribute of ParamAttr, gru_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        dtype(string): data type used in this unit

    Examples:

        .. code-block:: python

            import paddle.fluid.layers as layers
            from paddle.fluid.contrib.layers import BasicGRUUnit

            input_size = 128
            hidden_size = 256
            input = layers.data( name = "input", shape = [-1, input_size], dtype='float32')
            pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')

            gru_unit = BasicGRUUnit( "gru_unit", hidden_size )

            new_hidden = gru_unit( input, pre_hidden )

    """

    def __init__(self,
                 name_scope,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype='float32'):
        super(BasicGRUUnit_, self).__init__(name_scope, dtype)

        self._name = name_scope
        self._hiden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._dtype = dtype

    def _build_once(self, input, pre_hidden):
        self._input_size = input.shape[-1]
        assert (self._input_size > 0)

        self._gate_weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._hiden_size, 2 * self._hiden_size],
            dtype=self._dtype)

        self._candidate_weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._hiden_size, self._hiden_size],
            dtype=self._dtype)

        self._gate_bias = self.create_parameter(
            self._bias_attr,
            shape=[2 * self._hiden_size],
            dtype=self._dtype,
            is_bias=True)
        self._candidate_bias = self.create_parameter(
            self._bias_attr,
            shape=[self._hiden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, pre_hidden):
        xu_t, xr_t, xc_t = layers.split(input, num_or_sections=3, dim=-1)
        gate_input = layers.matmul(x=pre_hidden, y=self._gate_weight)
        gate_input = layers.elementwise_add(gate_input, self._gate_bias)
        hu_t, hr_t = layers.split(gate_input, num_or_sections=2, dim=-1)  
        u_add = layers.elementwise_add(xu_t, hu_t)
        r_add = layers.elementwise_add(xr_t, hr_t)
        u = self._gate_activation(u_add)
        r = self._gate_activation(r_add)
        r_hidden = r * pre_hidden
        candidate = layers.matmul(r_hidden, self._candidate_weight)
        candidate = layers.elementwise_add(xc_t, candidate)
        candidate = layers.elementwise_add(candidate, self._candidate_bias)
        c = self._activation(candidate)
        new_hidden = (1 - u) * pre_hidden + u * c

        return new_hidden

class GRUCell(RNNCell):
    """
    Gated Recurrent Unit cell. It is a wrapper for 
    `fluid.contrib.layers.rnn_impl.BasicGRUUnit` to make it adapt to RNNCell.

    The formula used is as follow:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}

    For more details, please refer to  `Learning Phrase Representations using
    RNN Encoder Decoder for Statistical Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_

    Examples:

        .. code-block:: python

            import paddle.fluid.layers as layers
            cell = layers.GRUCell(hidden_size=256)
    """

    def __init__(self,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype="float32",
                 name="GRUCell_"):
        """
        Constructor of GRUCell.

        Parameters:
            hidden_size (int): The hidden size in the GRU cell.
            param_attr(ParamAttr, optional): The parameter attribute for the learnable
                weight matrix. Default: None.
            bias_attr (ParamAttr, optional): The parameter attribute for the bias
                of GRU. Default: None.
            gate_activation (function, optional): The activation function for :math:`act_g`.
                Default: `fluid.layers.sigmoid`.
            activation (function, optional): The activation function for :math:`act_c`.
                Default: `fluid.layers.tanh`.
            dtype(string, optional): The data type used in this cell. Default float32.
            name(string, optional) : The name scope used to identify parameters and biases.
        """
        self.hidden_size = hidden_size
        self.gru_unit = BasicGRUUnit_(
            name, hidden_size, param_attr, bias_attr, gate_activation,
            activation, dtype)

    def call(self, inputs, states):
        """
        Perform calculations of GRU.

        Parameters:
            inputs(Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32.
            states(Variable): A tensor with shape `[batch_size, hidden_size]`.
                corresponding to :math:`h_{t-1}` in the formula. The data type
                should be float32.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` and \
                `new_states` is the same tensor shaped `[batch_size, hidden_size]`, \
                corresponding to :math:`h_t` in the formula. The data type of the \
                tensor is same as that of `states`.        
        """
        new_hidden = self.gru_unit(inputs, states)
        return new_hidden, new_hidden

    @property
    def state_shape(self):
        """
        The `state_shape` of GRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to :math:`h_{t-1}`.
        """
        return [self.hidden_size]