import tensorflow as tf


def QuantilizeFn(Wbit, Abit):
    def RoundPower2(x, k=4):
        bound = tf.pow(2.0, k - 1)
        min_val = tf.pow(2.0, -bound + 1.0)
        s = tf.sign(x)
        x = tf.clip_by_value(tf.abs(x), min_val, 1.0)
        p = tf.round(tf.log(x) / tf.log(2.))
        return s * tf.pow(2.0, p)

    def CeilPower2(x):
        p = tf.ceil(tf.log(x) / tf.log(2.))
        return tf.pow(2.0, p)

    @tf.custom_gradient
    def QuantilizeWeight(w):
        w = tf.clip_by_value(w, -1, 1)
        if Wbit == 1:   # BNN
            # mean = tf.reduce_mean(tf.abs(x))
            # E = tf.stop_gradient(mean)
            # return tf.sign(x) * E
            output = tf.sign(w)
        elif Wbit == 32:
            output = w
        else:   # QNN
            max = CeilPower2(tf.reduce_max(tf.abs(w)))
            w = w / max
            output =  max * RoundPower2(w, Wbit)

        def Grad(dy):
            return dy

        return output, Grad

    @tf.custom_gradient
    def QuantilizeActivation(x):
        if Abit == 1:   # BNN
            # mean = tf.reduce_mean(tf.abs(x))
            # E = tf.stop_gradient(mean)
            # return tf.sign(x) * E
            output = tf.sign(x)
        elif Abit == 32:
            output = x
        else:   # QNN
            max = CeilPower2(tf.reduce_max(tf.abs(x)))
            x = x / max
            output = max * RoundPower2(x, Abit)

        def Grad(dy):
            return dy

        return output, Grad

    return QuantilizeWeight, QuantilizeActivation