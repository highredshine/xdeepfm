from model_pipeline.models.linear import Linear
import tensorflow as tf
from tensorflow.keras.layers import add, Dense, Embedding, Layer, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding, StringLookup
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC


class CIN(Layer):
    """
    input_shape = (batch_size, m, D) 
        base embedding layer (H^0) of shape 
    m = number of fields in input data (units in base embedding layer)
    D = embedding size (embedding size has no effect on CIN layer sizes)
    output shape: (batch_size, 1)
    """
    def __init__(self, units, l2):
        """
        units = list of layer sizes (number of neuron units per layer)
        depth = (highest order of interaction - 1) # because first-order is omitted
        depth should be the length of the units
        """
        self.units = units
        self.depth = len(self.units)
        self.l2 = l2
        self.m, self.D = 0, 0
        self.w, self.b, self.activation_layers = [], [], []
        super(CIN, self).__init__()

    def build(self, input_shape):
        _, self.m, self.D = input_shape
        for k in range(self.depth):
            num_maps = self.units[k] # number of feature maps (units per layer, i.e. H^k)
            prev_num_maps = self.units[k - 1] if k > 0 else self.m # previous layer's size (i.e. H^(k-1))
            # parameter matrix for k-th hidden layer
            self.w.append(
                self.add_weight(
                    name = 'filter_{}-order'.format(k + 2),
                    shape = (1, prev_num_maps * self.m, # (H^(k-1) * m)
                                num_maps), # H^k parameter matrices for the k-th layer.
                    initializer = "random_normal",
                    regularizer = l2(self.l2),
                    trainable = True))
            self.b.append(
                self.add_weight(
                    name = 'bias_{}-order'.format(k + 2),
                    shape = (num_maps),
                    initializer = "zeros",
                    trainable = True))
        super(CIN, self).build(input_shape)
            
    def call(self, inputs):
        logit = []
        # we slide the filter across the tensor along the embedding dimension (D) 
        # thus, we split original feature matrix X^0 into base tensor Z^0
        base_tensor = tf.expand_dims(tf.unstack(inputs, axis = -1), axis = -1) # shape: (D, batch_size, m)
        curr_tensor = None
        for k in range(self.depth):
            # prev_tensor = tensor for the k-1 order layer, i.e. Z^(k-1) from X^(k-1)
            prev_tensor = tf.expand_dims(tf.unstack(curr_tensor, axis = -1), axis = -1) if k > 0 else base_tensor # shape: (D, batch_size, H^(k-1))
            # curr_tensor Z^k is the outer product of Z^(k-1) and base tensor Z^0
            curr_tensor = tf.matmul(base_tensor, prev_tensor, transpose_b = True) # shape: (D, batch_size, m, H^(k-1))
            # reshape so we can apply filters
            prev_num_maps = self.units[k] if k > 0 else self.m
            curr_tensor = tf.reshape(curr_tensor, (self.D, -1, prev_num_maps * self.m)) # shape: (D, batch_size, H^(k-1)*m)
            curr_tensor = tf.transpose(curr_tensor, perm = [1, 0, 2]) # shape: (batch_size, D, H^(k-1)*m)
            # perform compression by applying weight as filter
            curr_tensor = tf.nn.conv1d(curr_tensor, self.w[k], 1,'VALID') # shape: (batch_size, D, H^k)
            curr_tensor = tf.nn.bias_add(curr_tensor, self.b[k])
            curr_tensor = tf.transpose(curr_tensor, perm = [0, 2, 1]) # shape: (batch_size, H^k, D)
            curr_tensor, _ = tf.linalg.normalize(curr_tensor, axis = [-2,-1])
            # compress each tensor to logit
            logit.append(curr_tensor) # collect for all depth (order of interaction)
        pooling = tf.reduce_sum(tf.concat(logit, 1), -1) # shape: (batch_size, sum of all H^k for each k-th layer)
        
        return pooling
        

class Model(tf.keras.Model):
    def __init__(self, config, num_fields, cat_fields, vocabs):
        super(Model, self).__init__()
        params = config["model_params"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        # init setting
        self.num_fields = num_fields
        self.cat_fields = cat_fields
        self.m = len(num_fields) + len(cat_fields)
        self.D = params["embedding_dim"]
        # embedding: (batch_size, fields) -> (batch_size, embedding_dim(D))
        self.num_embedding_layers = [
            Dense(
                units = self.D,
                kernel_initializer = "random_uniform",
                kernel_regularizer = l2(params["l2"])
            ) for _ in num_fields
        ]
        self.cat_embedding_layers = [(
                StringLookup( # string indexer
                    vocabulary = vocabs[field],
                    num_oov_indices = 1,
                    mask_token = None,
                    encoding = 'utf-8'
                ),
                CategoryEncoding( # sparse one-hot encoding for linear input
                    max_tokens = len(vocabs[field]) + 1,
                    output_mode = "binary",
                    # sparse = True
                ),
                Embedding( # dense embedding for cin/dnn input
                    input_dim = len(vocabs[field]) + 1,
                    output_dim = self.D,
                    embeddings_initializer = "random_uniform",
                    embeddings_regularizer = l2(params["l2"])
                )
            ) for field in cat_fields
        ]
        # linear: (batch_size, sparse_features) -> (batch_size, 1)
        self.linear_layer = Linear(
            units = 1,
            regularizer = l2(params["l2"])
        )
        # CIN : (batch_size, m, D) -> (batch_size, 1)
        self.cin_layer = CIN(
            units = params["CIN_units"], 
            l2 = params["l2"])
        self.cin_output_layer = Dense(
            units = 1,
            use_bias = False,
            kernel_initializer = "random_uniform",
            kernel_regularizer = l2(params["l2"])
        )
        # DNN : (batch_size, m, D) -> (batch_size, 1)
        self.dnn_layer = [
            Dense(
                units = units,
                activation = params["DNN_activation"],
                kernel_initializer = "random_normal",
                kernel_regularizer = l2(params["l2"])
            ) for units in params["DNN_units"]
        ] 
        self.dnn_output_layer = Dense(
            units = 1,
            use_bias = False,
            kernel_initializer = "random_uniform",
            kernel_regularizer = l2(params["l2"])
        )
        # classification
        self.prediction_layer = Dense(
            units = 1,
            activation = 'sigmoid'
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            params["learning_rate"],
            decay_steps = params["steps"],
            decay_rate = params["decay_rate"],
            staircase=True
        )
        self.compile(
            optimizer = Adam(learning_rate = lr_schedule, clipnorm = 1),
            loss = BinaryCrossentropy(),
            metrics = [
                AUC(
                    num_thresholds = params["thresholds"],
                    summation_method = 'majoring'
                )
            ]
        )
    
    def call(self, inputs):
        def fill_nan(tensor):
            return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
        def concat(inputs, tensor, tensor_type = "dense"):
            if inputs == None:
                return tensor
            else:
                if tensor_type == "sparse":
                    return tf.sparse.concat(axis = 1, sp_inputs = [inputs, tensor])
                elif tensor_type == "dense":
                    return tf.concat(values = [inputs, tensor], axis = 1)
        # 1. preprocessing layers
        sparse_inputs, dense_inputs = None, None
        # transform numeric features
        for field, embedding_layer in zip(self.num_fields, self.num_embedding_layers):
            input_tensor = fill_nan(inputs[field])
            sparse_inputs = concat(sparse_inputs, input_tensor)
            dense_tensor = tf.expand_dims(embedding_layer(input_tensor), axis = 1)
            dense_inputs = concat(dense_inputs, dense_tensor)
        # transform categorical features
        for field, embedding_layer in zip(self.cat_fields, self.cat_embedding_layers):
            string_lookup, category_encoding, embedding = embedding_layer
            indices = string_lookup(inputs[field])
            encoding = fill_nan(category_encoding(indices))
            embedding = fill_nan(embedding(indices))
            sparse_inputs = concat(sparse_inputs, encoding)
            dense_inputs = concat(dense_inputs, embedding)
        # 3. linear
        linear_logit = self.linear_layer(sparse_inputs)
        # 5. cin
        cin_output = self.cin_layer(dense_inputs)
        cin_logit = self.cin_output_layer(cin_output)
        # 6. dnn
        _, m, D = dense_inputs.get_shape()
        dnn_output = tf.reshape(dense_inputs, (self.batch_size, m * D))
        for layer in self.dnn_layer:
            dnn_output = layer(dnn_output)
            dnn_output, _ = tf.linalg.normalize(dnn_output, axis = -1)
        dnn_logit = self.dnn_output_layer(dnn_output)
        # 4. final output
        logit = add([
            linear_logit,
            cin_logit, 
            dnn_logit
        ])
        logit = fill_nan(logit)
        probabilities = tf.clip_by_value(self.prediction_layer(logit), 0, 1)
        return probabilities
    