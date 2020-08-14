import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers, Model


weight_decay = 0.0005

inputs = tf.keras.Input(shape=(32,32,3))
x = Conv2D(
        16, 3, padding = 'same', use_bias = False, 
        kernel_regularizer = regularizers.l2(weight_decay))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Block 1, channels: 16
# Block 1, unit 1
shortcut = Conv2D(
        16, 1, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
shortcut = BatchNormalization()(shortcut)

x = Conv2D(
        16, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        16, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

# Block 1, unit 2
shortcut = x

x = Conv2D(
        16, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        16, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

# Block 1, unit 3
shortcut = x

x = Conv2D(
        16, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        16, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)


# Block 2, channels: 32
# Block 2, unit 1
shortcut = Conv2D(
        32, 1, strides = 2, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
shortcut = BatchNormalization()(shortcut)

x = Conv2D(
        32, 3, strides = 2, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        32, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

# Block 2, unit 2
shortcut = x

x = Conv2D(
        32, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        32, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

# Block 2, unit 3
shortcut = x

x = Conv2D(
        32, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        32, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

# Block 3, channels: 64
# Block 3, unit 1
shortcut = Conv2D(
        64, 1, strides = 2, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
shortcut = BatchNormalization()(shortcut)

x = Conv2D(
        64, 3, strides = 2, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        64, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

# Block 3, unit 2
shortcut = x

x = Conv2D(
        64, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        64, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

# Block 3, unit 3
shortcut = x

x = Conv2D(
        64, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(
        64, 3, padding = 'same', use_bias = False,
        kernel_regularizer = regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x += shortcut
x = Activation('relu')(x)

x = MaxPooling2D(pool_size = 8)(x)
x = Flatten()(x)
x = Dense(10, kernel_regularizer = regularizers.l2(weight_decay))(x)
outputs = Activation('softmax')(x)

model = Model(inputs = inputs, outputs = outputs)
