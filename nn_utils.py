import tensorflow as tf
from tensorflow.keras import regularizers
from quantization import QuantilizeFn 

BITW = 4
BITA = 4
QuantilizeWeight, QuantilizeActivation = QuantilizeFn(BITW, BITA)

class QConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_depth,
            kernel_size,
            strides = [1, 1],
            padding = 'SAME', 
            quantilize = False,
            weight_decay = 0.0005,
            use_bias = True,
            name = None):

        super(QConv2D, self).__init__()
        self.kernel_depth = kernel_depth
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        # self.name = name
        self.quantilize = quantilize
        self.weight_decay = weight_decay
        self.use_bias = use_bias

    def build(self, input_shape):
        # filters_name = self.name + '/weights'
        self.filters = self.add_weight(
                # filters_name,
                shape = [ 
                    self.kernel_size,
                    self.kernel_size, 
                    int(input_shape[-1]),                   
                    self.kernel_depth],
                initializer = tf.keras.initializers.GlorotUniform(),
                regularizer = regularizers.l2(self.weight_decay))

        if self.quantilize:
            self.filters = QuantilizeWeight(self.filters)

        if self.use_bias:
            # bias_name = self.name + '/bias'
            self.bias = self.add_weight(
                    # bias_name,
                    shape = self.kernel_depth,
                    initializer = tf.keras.initializers.Zeros())
        
    def call(self, input_tensor):
        output = tf.nn.conv2d(
                input_tensor, self.filters, self.strides, self.padding)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return output
