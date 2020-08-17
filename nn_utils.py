import tensorflow as tf
from tensorflow.keras import regularizers
from quantization import QuantilizeFn, tangent
import pdb

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
        self.quantilize = quantilize
        self.weight_decay = weight_decay
        self.use_bias = use_bias

    def build(self, input_shape):
        self.filters = self.add_weight(
                shape = [ 
                    self.kernel_size,
                    self.kernel_size, 
                    int(input_shape[-1]),                   
                    self.kernel_depth],
                initializer = tf.keras.initializers.GlorotUniform(),
                regularizer = regularizers.l2(self.weight_decay))

        # if self.quantilize == 'ste':
            # self.filters = QuantilizeWeight(self.filters)
        # elif self.quantize == 'ng':
            # filters_quantilize = QuantilizeWeight(self.filters)
            # self.filters = (self.filters, filters_quantilize, alpha)

        # if self.quantilize:
            # # self.filters = tf.clip_by_value(self.filters, -1, 1)
            # self.filters = QuantilizeWeight(self.filters)

        if self.use_bias:
            self.bias = self.add_weight(
                    shape = self.kernel_depth,
                    initializer = tf.keras.initializers.Zeros())
        
    def call(self, input_tensor):
        if self.quantilize:
            filters = tf.clip_by_value(self.filters, -1, 1)
            filters = QuantilizeWeight(filters)
            input_tensor = QuantilizeActivation(input_tensor)
        else:
            filters = self.filters

        output = tf.nn.conv2d(
                input_tensor, filters, self.strides, self.padding)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return output
