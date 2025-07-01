import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''
    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        layer_output = op(self, layer_input, *args, **kwargs)
        self.layers[name] = layer_output
        self.feed(layer_output)
        return self
    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, n_classes=20, keep_prob=1):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = 1.0
        self.setup(is_training, n_classes, keep_prob)

    def setup(self, is_training, n_classes, keep_prob):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.01), name=name, trainable=self.trainable)

    def make_w_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        stddev=0.01
        return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev), name=name, trainable=self.trainable)

    def make_b_var(self, name, shape):
        return tf.Variable(tf.zeros(shape), name=name, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.shape[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        kernel = self.make_w_var(f'{name}_weights', shape=[k_h, k_w, c_i // group, c_o])
        if group == 1:
            # This is the common-case. Convolve the input without any further complications.
            output = tf.nn.conv2d(input, kernel, strides=[1, s_h, s_w, 1], padding=padding)
        else:
            # Split the input into groups and then convolve each of them independently
            input_groups = tf.split(input, group, axis=3)
            kernel_groups = tf.split(kernel, group, axis=3)
            output_groups = [tf.nn.conv2d(i, k, strides=[1, s_h, s_w, 1], padding=padding) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            output = tf.concat(output_groups, axis=3)
        # Add the biases
        if biased:
            biases = self.make_b_var(f'{name}_biases', [c_o])
            output = tf.nn.bias_add(output, biases)
        if relu:
            # ReLU non-linearity
            output = tf.nn.relu(output, name=name)
        return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.shape[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        kernel = self.make_w_var(f'{name}_weights', shape=[k_h, k_w, c_i // group, c_o])
        if group == 1:
            # This is the common-case. Convolve the input without any further complications.
            output = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding=padding, dilations=[dilation, dilation])
        else:
            # Split the input into groups and then convolve each of them independently
            input_groups = tf.split(input, group, axis=3)
            kernel_groups = tf.split(kernel, group, axis=3)
            output_groups = [tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding=padding, dilations=[dilation, dilation]) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            output = tf.concat(output_groups, axis=3)
        # Add the biases
        if biased:
            biases = self.make_b_var(f'{name}_biases', [c_o])
            output = tf.nn.bias_add(output, biases)
        if relu:
            # ReLU non-linearity
            output = tf.nn.relu(output, name=name)
        return output
        
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool2d(input,
                              ksize=[k_h, k_w],
                              strides=[s_h, s_w],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool2d(input,
                              ksize=[k_h, k_w],
                              strides=[s_h, s_w],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        input_shape = input.shape
        if len(input_shape) > 2:
            # The input is spatial. Vectorize it first.
            dim = 1
            for d in input_shape[1:]:
                dim *= d
            feed_in = tf.reshape(input, [-1, dim])
        else:
            feed_in, dim = (input, input_shape[-1])
        weights = self.make_var('weights', shape=[dim, num_out])
        biases = self.make_var('biases', [num_out])
        x = tf.matmul(feed_in, weights) + biases
        if relu:
            x = tf.nn.relu(x, name=name)
        return x

    @layer
    def softmax(self, input, name):
        input_shape = input.shape
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, axis=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)
        
    @layer
    def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
        bn = tf.keras.layers.BatchNormalization(scale=scale)
        x = bn(input, training=is_training)
        if activation_fn is not None:
            x = activation_fn(x)
        return x

    @layer
    def dropout(self, input, keep_prob, name):
        rate = 1 - keep_prob
        return tf.keras.layers.Dropout(rate)(input)

    @layer
    def upsample(self, input, size_h, size_w, name):
        return tf.image.resize(input, size=[size_h, size_w])

    @layer
    def pyramid_pooling(self, input, o_c, pool_size, name):
        dims = tf.shape(input)
        out_height, out_width = dims[1], dims[2]
        pool_ly = tf.nn.avg_pool2d(input, ksize=[pool_size, pool_size], strides=[pool_size, pool_size],
                                     padding=DEFAULT_PADDING, name='pool_ly')
        weight = self.make_w_var(f'{name}_weights', shape=[3, 3, pool_ly.shape[-1], o_c])
        biases = self.make_var('biases', o_c)
        conv_ly = tf.nn.conv2d(pool_ly, weight, strides=[1, 1, 1, 1], padding='SAME', name='conv_ly')
        conv_ly = tf.nn.bias_add(conv_ly, biases)
        conv_ly = tf.nn.relu(conv_ly, name='relu_ly')
        output = tf.image.resize(conv_ly, [out_height, out_width], method='bilinear')
        return output
