import tensorflow as tf
from meta_policy_search.utils.utils import get_original_tf_name, get_last_scope
import tensorflow_probability as tfp
tfd = tfp.distributions
import os


def output_weights(model_out,fast_weights):
    j=0
    #print('len_fast_weights=',len(fast_weights))
    for i, layer in enumerate(model_out.layers):
        print(i,layer)
        print('j=',j)
        #print(layer.kernel_posterior)  #  don't delete, very important
        try:
            layer.kernel_posterior =  tfd.Independent(tfd.Normal(loc=fast_weights[j],scale=tf.math.softplus(fast_weights[j+1])) ,reinterpreted_batch_ndims=len(layer.kernel_posterior.mean().shape))
            layer.bias_posterior =  tfd.Independent(tfd.Deterministic(loc=fast_weights[j+2]) ,reinterpreted_batch_ndims=1)
            j+=3
            print('tfp')
        except AttributeError:
            continue


def construct_fc_weights(output_dim, hidden_sizes, hidden_nonlinearity, output_nonlinearity, input_var):

    x = tf.keras.layers.Input(shape=input_var.get_shape())
    inp = x
    for idx, hidden_size in enumerate(hidden_sizes):
        x = tfp.layers.DenseReparameterization(hidden_size ,activation=hidden_nonlinearity)(x)
    out_var = tfp.layers.DenseReparameterization(output_dim,activation=output_nonlinearity )(x)
    model = tf.keras.Model(inputs=inp, outputs=out_var)

    return model


top_scope = tf.get_variable_scope()
#print('top_scope=',top_scope)

model_vars = {}


def get_vars(dict):
    var_scope = list(dict.keys())[0]
    names = list(dict.values())[0]
    with tf.variable_scope(top_scope):
        print('top_scope=',top_scope)
        scope = tf.get_default_graph().get_name_scope()
        print('re_top_scope=',scope)
        with tf.variable_scope(var_scope,reuse=True):
            scope = tf.get_default_graph().get_name_scope()
            print('reused_scope=',scope)
            params = [tf.get_variable(var_name) for var_name in names]

    return params


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
    global model_vars

    assert input_var is not None or input_dim is not None

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')

    with tf.variable_scope(name):
        model = construct_fc_weights( output_dim=output_dim,
                                                 hidden_sizes=hidden_sizes,
                                                 hidden_nonlinearity=hidden_nonlinearity,
                                                 output_nonlinearity=output_nonlinearity,
                                                 input_var=input_var
                                                 )


    #with tf.variable_scope(top_scope):
        #with tf.variable_scope(name,auxiliary_name_scope=False):
        if not reuse:
            #model_vars = tf.RaggedTensor(model.trainable_variables,name='model_vars')
            scope = tf.get_default_graph().get_name_scope()
            print('scope=',scope)
            var_names = [ os.path.relpath(x.name, scope) for x in  model.trainable_variables]
            print('var_names',var_names)
            model_vars[scope] = var_names
            print('model_vars',model_vars)

        else:
            tf.get_variable_scope().reuse_variables()   #  set reuse=True
            print('reuse model_vars')

            print('model_vars',model_vars)
            '''
            with tf.variable_scope(top_scope):
                print('top_scope=',top_scope)
                scope = tf.get_default_graph().get_name_scope()
                print('re_top_scope=',scope)
                with tf.variable_scope(list(model_vars.keys())[0]):
                    scope = tf.get_default_graph().get_name_scope()
                    print('reused_scope=',scope)

                    params = [tf.get_variable(var_name) for var_name in list(model_vars.values())[0]]
                    print('retrived params=',params)
            '''
            get_vars(model_vars)
            print('retrived params=',params)
            output_weights(model, params)

        output_var = model(input_var)


        # check trainable_variables
        '''
        current_scope = tf.get_default_graph().get_name_scope()
        print('current_scope=',current_scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
        print('trainable_vars=',trainable_vars)
        '''
    '''
    with tf.variable_scope(name,auxiliary_name_scope=False):
        current_scope = tf.get_default_graph().get_name_scope()
        print('current_scope=',current_scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
        print('trainable_vars=',trainable_vars)
        ###########################
    '''

    return input_var, output_var

'''
def output_layer(layer,fast_weights):
    j=0
    layer.kernel_posterior =  tfd.Independent(tfd.Normal(loc=fast_weights[j],scale=tf.math.softplus(fast_weights[j+1])) ,reinterpreted_batch_ndims=len(layer.kernel_posterior.mean().shape))
    layer.bias_posterior =  tfd.Independent(tfd.Deterministic(loc=fast_weights[j+2]) ,reinterpreted_batch_ndims=1)
'''




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
    print("forward_mlp_params=",mlp_params)

    model = construct_fc_weights( output_dim=output_dim,
                                             hidden_sizes=hidden_sizes,
                                             hidden_nonlinearity=hidden_nonlinearity,
                                             output_nonlinearity=output_nonlinearity,
                                             input_var=input_var
                                             )
    print("model.trainable_variables=",model.trainable_variables)
    output_weights(model, list(mlp_params.values()))
    output_var = model(input_var)
    print('forward_params',list(mlp_params.values()))
    print('forward_output_var=',output_var)
    print('check_forward_grads=',tf.gradients(output_var,list(mlp_params.values())))


    return input_var, output_var # Todo why return input_var?
