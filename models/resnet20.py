import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

import pdb


class ResnetUnitL2(tf.keras.layers.Layer):
# class ResnetUnitL2(tf.keras.Model):
    def __init__(
            self, 
            outputs_depth, 
            strides = 1, 
            is_first = False, 
            weight_decay = 0.0005):

        super(ResnetUnitL2, self).__init__()
        self.is_first = is_first

        self.conv2a = Conv2D( 
                outputs_depth, 3, strides, padding = 'same', use_bias = False,
                kernel_regularizer = regularizers.l2(weight_decay))
        self.bn2a = BatchNormalization()
        
        self.conv2b = Conv2D(
                outputs_depth, 3, padding='same', use_bias = False,
                kernel_regularizer = regularizers.l2(weight_decay))
        self.bn2b = BatchNormalization()

        if is_first:
            self.conv_shortcut = Conv2D(
                    outputs_depth, 1, strides, 
                    padding = 'same', use_bias = False,
                    kernel_regularizer = regularizers.l2(weight_decay))
            self.bn_shortcut = BatchNormalization()

    def call(self, input_tensor):
        if self.is_first:
            shortcut = self.conv_shortcut(input_tensor)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = input_tensor

        x = self.conv2a(input_tensor)
        x = self.bn2a(x)
        x = Activation('relu')(x)

        x = self.conv2b(x)
        x = self.bn2b(x)

        x += shortcut
        return Activation('relu')(x)


class ResnetBlockL2(tf.keras.layers.Layer):
# class ResnetBlockL2(tf.keras.Model):
    def __init__(
            self,
            num_units,
            outputs_depth,
            strides,
            weight_decay = 0.0005):

        super(ResnetBlockL2, self).__init__()
        self.num_units = num_units
        self.units = []
        self.units.append(
                ResnetUnitL2(
                    outputs_depth, strides = strides, is_first = True, 
                    weight_decay = weight_decay))
        for i in range(1, self.num_units):
            self.units.append(
                    ResnetUnitL2( 
                        outputs_depth, weight_decay = weight_decay))

    def call(self, input_tensor):
        x = self.units[0](input_tensor)
        for i in range(1, self.num_units):
            x = self.units[i](x)

        return x


class Resnet20(tf.keras.Model):
    def __init__(
            self,
            weight_decay,
            class_num):

        super(Resnet20, self).__init__(name = '')
        self.conv_first = Conv2D(
                16, 3, padding = 'same', use_bias = False,
                kernel_regularizer = regularizers.l2(weight_decay))
        self.bn_first = BatchNormalization()
        self.dense = Dense(
                class_num, kernel_regularizer = regularizers.l2(weight_decay))
        self.block1 = ResnetBlockL2(3, 16, 1, weight_decay = weight_decay)
        self.block2 = ResnetBlockL2(3, 32, 2, weight_decay = weight_decay)
        self.block3 = ResnetBlockL2(3, 64, 2, weight_decay = weight_decay)

    def call(self, input_tensor):
        x = self.conv_first(input_tensor)
        x = self.bn_first(x)
        x = Activation('relu')(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = MaxPooling2D(pool_size = 8)(x)
        x = Flatten()(x)
        x = self.dense(x)

        return Activation('softmax')(x)

weight_decay = 0.0005
class_num = 10

model = Resnet20(weight_decay, class_num)


# inputs = tf.keras.Input(shape=(32,32,3))
# x = Conv2D(
        # 16, 3, padding = 'same', use_bias = False, 
        # kernel_regularizer = regularizers.l2(weight_decay))(inputs)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)

# # Block 1, channels: 16
# # Block 1, unit 1
# shortcut = Conv2D(
        # 16, 1, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# shortcut = BatchNormalization()(shortcut)

# x = Conv2D(
        # 16, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 16, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# # Block 1, unit 2
# shortcut = x

# x = Conv2D(
        # 16, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 16, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# # Block 1, unit 3
# shortcut = x

# x = Conv2D(
        # 16, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 16, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)


# # Block 2, channels: 32
# # Block 2, unit 1
# shortcut = Conv2D(
        # 32, 1, strides = 2, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# shortcut = BatchNormalization()(shortcut)

# x = Conv2D(
        # 32, 3, strides = 2, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 32, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# # Block 2, unit 2
# shortcut = x

# x = Conv2D(
        # 32, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 32, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# # Block 2, unit 3
# shortcut = x

# x = Conv2D(
        # 32, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 32, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# # Block 3, channels: 64
# # Block 3, unit 1
# shortcut = Conv2D(
        # 64, 1, strides = 2, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# shortcut = BatchNormalization()(shortcut)

# x = Conv2D(
        # 64, 3, strides = 2, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 64, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# # Block 3, unit 2
# shortcut = x

# x = Conv2D(
        # 64, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 64, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# # Block 3, unit 3
# shortcut = x

# x = Conv2D(
        # 64, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(
        # 64, 3, padding = 'same', use_bias = False,
        # kernel_regularizer = regularizers.l2(weight_decay))(x)
# x = BatchNormalization()(x)
# x += shortcut
# x = Activation('relu')(x)

# x = MaxPooling2D(pool_size = 8)(x)
# x = Flatten()(x)
# x = Dense(10, kernel_regularizer = regularizers.l2(weight_decay))(x)
# outputs = Activation('softmax')(x)

# model = Model(inputs = inputs, outputs = outputs)
