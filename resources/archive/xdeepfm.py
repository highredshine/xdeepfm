from tensorflow.keras.initializers import Zeros, GlorotNormal, GlorotUniform
from tensorflow.keras.layers import Activation, Add, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import tensorflow as tf


def set_data_dir(path, model_name):
    global DATA_DIR, INPUT_PATH, CACHE_DIR, WEIGHT_PATH
    DATA_DIR = path
    CACHE_DIR = DATA_DIR + "/cache/" + model_name
    INPUT_PATH = CACHE_DIR + "/input.parquet"
    WEIGHT_PATH = CACHE_DIR + "/weights.h5"


class Linear(Layer):
    """
    input: (batch_size, feature_count)
    weight: (feature_count, 1)
    output: (batch_size, 1)
    """

    def __init__(self, l2 = 0.0001, **kwargs):
        self.l2 = l2
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        print("buliding linear layer...")
        self.feature_count = input_shape[-1]
        self.w = self.add_weight(name = 'linear_kernel',
                                shape = (self.feature_count, 1),
                                initializer = GlorotNormal(),
                                regularizer = l2(self.l2),
                                trainable = True)
        self.b = self.add_weight(name = 'linear_bias',
                                shape = (1, ),
                                initializer = Zeros(),
                                trainable = True)
        super(Linear, self).build(input_shape)
    
    def call(self, inputs, training = None, **kwargs):
        outputs = tf.tensordot(inputs, self.w, axes = (-1, 0)) # shape: (batch_size, 1)
        outputs = tf.nn.bias_add(outputs, self.b)
        return outputs


class Embedding(Layer):
    """
    input shape : (batch_size, feature_count)
    feature_count : number of features

    output shape: (batch_size, m, D)
    m : number o fields
    D : embedding dimension
    """
    def __init__(self, D = 10, **kwargs):
        self.D = D
        # self.numeric_embeddings = []
        # self.categorical_embeddings = []
        self.embeddings = []
        super(Embedding, self).__init__(**kwargs)
    
    def set_metadata(self, metadata):
        self.m = metadata["num_fields"]
        self.num_features = metadata["num_features"]
        self.numeric_fields = metadata["numeric_fields"]
        self.categorical_fields = metadata["categorical_fields"]
        self.cardinalities = metadata["cardinalities"]
    
    def build(self, input_shape):
        print("buliding Embedding layer...")
        # create embedding weight vectors for numerical features
        for field in self.numeric_fields:
            self.embeddings.append(self.add_weight(name = 'numeric_{}'.format(field),
                                    shape = (1, self.D),
                                    initializer = GlorotNormal(),
                                    trainable = True))
        # create embeddnig layer for categorical features
        for field in self.categorical_fields:
            cardinality = self.cardinalities[field]
            self.embeddings.append(self.add_weight(name = 'categorical_{}'.format(field),
                                    shape = (cardinality, self.D),
                                    initializer = GlorotNormal(),
                                    trainable = True))
        super(Embedding, self).build(input_shape)

    def call(self, inputs, training = None, **kwargs):
        # split single sparse input vector into 'm' field vectors by corresponding cardinalities.
        field_vectors = tf.split(inputs, list(self.cardinalities.values()), axis = 1)
        embedded = []
        for i in range(self.m): # apply embedding to each field
            dense_vector = tf.tensordot(field_vectors[i], self.embeddings[i], axes = (-1, 0)) # (batch_size, D)
            embedded.append(dense_vector) # (m, batch_size, D)
        outputs = tf.transpose(embedded, perm = [1, 0, 2]) # (batch_size, m, D)
        return outputs


class CIN(Layer):
    """
    input_shape = (batch_size, m, D) 
        base embedding layer (H^0) of shape 
    m = number of fields in input data (units in base embedding layer)
    D = embedding size (embedding size has no effect on CIN layer sizes)

    output shape: (batch_size, 1)
    """

    def __init__(self, layer_units, activation = 'linear', l2 = 0.0001, **kwargs):
        """
        layer_units = list of layer sizes (number of neuron units per layer)
        depth = (highest order of interaction - 1) # because first-order is omitted
        depth should be the length of the layer_units
        """
        self.layer_units = layer_units
        self.depth = len(self.layer_units)
        self.activation = activation
        self.l2 = l2
        self.m, self.D = 0, 0
        self.w, self.b, self.activation_layers = [], [], []
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        print("buliding CIN...")
        _, self.m, self.D = input_shape
        for k in range(self.depth):
            num_maps = self.layer_units[k] # number of feature maps (units per layer, i.e. H^k)
            prev_num_maps = self.layer_units[k - 1] if k > 0 else self.m # previous layer's size (i.e. H^(k-1))
            # parameter matrix for k-th hidden layer
            self.w.append(self.add_weight(name = 'filter_{}-order'.format(k + 2),
                                        shape = (1, prev_num_maps * self.m, # (H^(k-1) * m)
                                                 num_maps), # H^k parameter matrices for the k-th layer.
                                        initializer = GlorotUniform(),
                                        regularizer = l2(self.l2),
                                        trainable = True))
            self.b.append(self.add_weight(name = 'bias_{}-order'.format(k + 2),
                                        shape = (num_maps),
                                        initializer = Zeros(),
                                        trainable = True))
            self.activation_layers.append(Activation(self.activation))
        # last regression layer
        self.output_units = Linear(l2 = self.l2)
        super(CIN, self).build(input_shape)
            
    def call(self, inputs, training = None, **kwargs):
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
            prev_num_maps = self.layer_units[k] if k > 0 else self.m
            curr_tensor = tf.reshape(curr_tensor, (self.D, -1, prev_num_maps * self.m)) # shape: (D, batch_size, H^(k-1)*m)
            curr_tensor = tf.transpose(curr_tensor, perm = [1, 0, 2]) # shape: (batch_size, D, H^(k-1)*m)
            # perform compression by applying weight as filter
            curr_tensor = tf.nn.conv1d(curr_tensor, self.w[k], 1,'VALID') # shape: (batch_size, D, H^k)
            curr_tensor = tf.nn.bias_add(curr_tensor, self.b[k])
            curr_tensor = tf.transpose(curr_tensor, perm = [0, 2, 1]) # shape: (batch_size, H^k, D)
            # compress each tensor to logit
            logit.append(curr_tensor) # collect for all depth (order of interaction)
        pooling = tf.reduce_sum(tf.concat(logit, 1), -1) # shape: (batch_size, sum of all H^k for each k-th layer)
        outputs = self.output_units(pooling) # shape: (batch_size, 1)
        return outputs


class DNN(Layer):
    """
    input shape : (batch_size, m, D)
    m : number of fields
    D : embedding dimension

    output shape: (batch_size, 1)
    """

    def __init__(self, layer_units, activation = 'relu', l2 = 0.0001, **kwargs):
        self.layer_units = layer_units
        self.depth = len(self.layer_units)
        self.activation = activation
        self.l2 = l2
        self.w, self.b, self.activation_layers = [], [], []
        super(DNN, self).__init__(**kwargs)
        
    def build(self, input_shape):
        """
        a plain DNN (deep neural network) is a multi-layer perceptron with hidden layers.
        """
        print("buliding DNN...")
        _, self.m, self.D = input_shape
        hidden_units = [self.m * self.D] + list(self.layer_units)
        for i in range(self.depth):
            self.w.append(self.add_weight(name = 'kernel{}'.format(i + 1),
                                    shape = (hidden_units[i], hidden_units[i + 1]),
                                    initializer = GlorotNormal(),
                                    regularizer = l2(self.l2),
                                    trainable = True))
            self.b.append(self.add_weight(name = 'bias{}'.format(i + 1),
                                    shape = (self.layer_units[i],),
                                    initializer = Zeros(),
                                    trainable = True))
            self.activation_layers.append(Activation(self.activation))
        self.output_units = Linear(l2 = self.l2)
        super(DNN, self).build(input_shape)    

    
    def call(self, inputs, training = None, **kwargs):
        logit = tf.reshape(inputs, (-1, self.m * self.D))
        for i in range(self.depth):
            logit = tf.tensordot(logit, self.w[i], axes = (-1, 0)) # shape: (batch_size, hidden_units[i])
            logit = tf.nn.bias_add(logit, self.b[i])
            logit = self.activation_layers[i](logit, training = training)
        outputs = self.output_units(logit) # shape: (batch_size, 1)
        return outputs


class BinaryClassificationLayer(Layer):
    """
    applying a sigmoid function on an output to perform binary classification
    """
    def __init__(self, **kwargs):
        self.b = None
        super(BinaryClassificationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("buliding output layer...")
        self.b = self.add_weight(name = 'prediction_bias',
                                shape = (1,),
                                initializer = Zeros())
        super(BinaryClassificationLayer, self).build(input_shape)

    def call(self, inputs, training = None, **kwargs):
        logit = inputs
        logit = tf.nn.bias_add(logit, self.b)
        logit = tf.sigmoid(logit)
        prediction = logit
        # prediction = tf.cast(tf.greater(logit, 0.5), dtype = tf.int64)
        return prediction


class Model(tf.keras.Model):
    """
    input shape: (batch_size * m)
        m : number of fields
        input tensor should be ordered by numeric fields and categorical fields.
    """

    def __init__(self):
        super(Model, self).__init__()
        ### Hyperparameters
        self.l2 = 0.0001
        # Embedding layer hyperparameters
        self.D = 10
        # CIN hyperparameters
        self.cin_layer_units = [200, 200, 200]
        self.cin_activation = 'linear'
        # Plain DNN hyperparameters
        self.dnn_layer_units = [400, 400]
        self.dnn_activation = 'relu'

        # Layers
        self.linear_layer = Linear(l2 = self.l2)
        self.embedding_layer = Embedding(D = self.D)
        self.cin_layer = CIN(
            layer_units = self.cin_layer_units,
            activation = self.cin_activation, l2 = self.l2)
        self.dnn_layer = DNN(
            layer_units = self.dnn_layer_units,
            activation = self.dnn_activation, l2 = self.l2)
        self.prediction_layer = BinaryClassificationLayer()

        # training hyperparameters
        self.learning_rate = 0.001
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.loss = BinaryCrossentropy()
        self.metric = AUC()
        self.batch_size = 4096
        self.epochs = 5

    def call(self, inputs, **kwargs):
        # input (sparse) shape: (batch_size, number of features)
        linear_logit = self.linear_layer(inputs) # output shape: (batch_size, 1)
        embedding_matrix = self.embedding_layer(inputs) # (batch_size, m, D) (dense)
        cin_logit = self.cin_layer(embedding_matrix)
        dnn_logit = self.dnn_layer(embedding_matrix)
        output = Add()([linear_logit, cin_logit, dnn_logit])
        prediction = self.prediction_layer(output)
        return prediction
    
    def set_metadata(self, metadata):
        self.embedding_layer.set_metadata(metadata)