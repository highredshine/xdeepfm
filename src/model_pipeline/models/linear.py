import tensorflow as tf
import tensorflow.keras as keras

class Linear(keras.layers.Layer):
    def __init__(self, 
        units, 
        regularizer, 
        activation = 'linear',
        sparse = False, 
        input_dim = None,
        use_bias = True
    ):
        super(Linear, self).__init__()
        self.units = units
        self.regularizer = regularizer
        self.activation = keras.activations.get(activation)
        self.sparse = sparse
        if self.sparse:
            self.input_dim = input_dim
        self.use_bais = use_bias

    def build(self, input_shape):
        if not self.sparse:
            self.input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape = (self.input_dim, self.units),
            initializer = "random_normal",
            regularizer = self.regularizer,
            trainable = True
        )
        if self.use_bais:
            self.b = self.add_weight(
                shape = (self.units,), 
                initializer = "zeros", 
                trainable = True
            )

    def call(self, inputs):
        if self.sparse:
            if self.use_bais:
                logit = tf.sparse.sparse_dense_matmul(inputs, self.w) + self.b
            else:
                logit = tf.sparse.sparse_dense_matmul(inputs, self.w)
        else:
            if self.use_bais:
                logit = tf.matmul(inputs, self.w) + self.b
            else:
                logit = tf.matmul(inputs, self.w)
        logit = self.activation(logit)
        return logit