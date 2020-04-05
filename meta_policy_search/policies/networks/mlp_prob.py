import tensorflow as tf
from meta_policy_search.utils.utils import get_original_tf_name, get_last_scope
import tensorflow_probability as tfp
tfd = tfp.distributions

def create_mlp(name,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               #b_init=tf.constant_initializer(0.0),
               reuse=False
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    with tf.variable_scope(name):
        x = input_var

        for idx, hidden_size in enumerate(hidden_sizes):
            '''
            x = tf.layers.dense(x,
                                hidden_size,
                                name='hidden_%d' % idx,
                                activation=hidden_nonlinearity,
                                kernel_initializer=w_init,
                                bias_initializer=b_init,
                                reuse=reuse,
                                )
            '''
            x = tfp.layers.DenseReparameterization(hidden_size ,activation=hidden_nonlinearity)(x)
        '''
        output_var = tf.layers.dense(x,
                                     output_dim,
                                     name='output',
                                     activation=output_nonlinearity,
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     reuse=reuse,
                                     )
        '''
        output_var = tfp.layers.DenseReparameterization(output_dim,activation=output_nonlinearity )(x)

    return input_var, output_var



def output_weights(model_out,fast_weights):
    j=0
    #print('len_fast_weights=',len(fast_weights))
    for i, layer in enumerate(model_out.layers):
        #print(i,layer)
        #print('j=',j)

        #print(layer.kernel_posterior)  #  don't delete, very important
        try:
            layer.kernel_posterior =  tfd.Independent(tfd.Normal(loc=fast_weights[j],scale=tf.math.softplus(fast_weights[j+1])) ,reinterpreted_batch_ndims=len(layer.kernel_posterior.mean().shape))
            layer.bias_posterior =  tfd.Independent(tfd.Deterministic(loc=fast_weights[j+2]) ,reinterpreted_batch_ndims=1)
            j+=3
        #print('tfp')
        except AttributeError:
            continue





def forward_mlp(output_dim,
                hidden_sizes,
                hidden_nonlinearity,
                output_nonlinearity,
                input_var,
                mlp_params,
                ):
    """
    Creates the forward pass of an mlp given the input vars and the mlp params. Assumes that the params are passed in
    order i.e. [hidden_0/kernel, hidden_0/bias, hidden_1/kernel, hidden_1/bias, ..., output/kernel, output/bias]
    Args:
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        mlp_params (OrderedDict): OrderedDict of the params of the neural network.

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """
    #print(mlp_params)
    inp_var = tf.keras.layers.Input(shape=input_var.get_shape())
    inp_var, out_var = create_mlp(name='mean_network_forward',
                                             output_dim=output_dim,
                                             hidden_sizes=hidden_sizes,
                                             hidden_nonlinearity=hidden_nonlinearity,
                                             output_nonlinearity=output_nonlinearity,
                                             input_var=inp_var
                                             )
    model = tf.keras.Model(inputs=inp_var, outputs=out_var)
    output_weights(model, list(mlp_params.values()))
    output_var = model(input_var)








    '''
    x = input_var
    idx = 0
    bias_added = False
    sizes = tuple(hidden_sizes) + (output_dim,)

    if output_nonlinearity is None:
        output_nonlinearity = tf.identity

    for name, param in mlp_params.items():
        assert str(idx) in name or (idx == len(hidden_sizes) and "output" in name)

        if "kernel" in name:
            assert param.shape == (x.shape[-1], sizes[idx])
            x = tf.matmul(x, param)
        elif "bias" in name:
            assert param.shape == (sizes[idx],)
            x = tf.add(x, param)
            bias_added = True
        else:
            raise NameError

        if bias_added:
            if "hidden" in name:
                x = hidden_nonlinearity(x)
            elif "output" in name:
                x = output_nonlinearity(x)
            else:
                raise NameError
            idx += 1
            bias_added = False
    output_var = x
    '''

    return input_var, output_var # Todo why return input_var?
