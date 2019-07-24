import tensorflow as tf

class scSE:
    
    def __init__(self, layer):
        self.layer = layer
        self.layer_shape = self.layer.get_shape().as_list()
        
 
    def cSE_weights_initializer(self):
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        cse_weights_dict = {}
        cse_weights_dict['1_by_1_conv'] = tf.Variable(conv_initializer(shape = [1, 1, self.layer_shape[-1], 1]))
        
        self.cse_weights_dict = cse_weights_dict
        
    
    def sSE_weights_initializer(self):
        initializer = tf.contrib.layers.xavier_initializer()
        sse_weights_dict = {}
        
        if self.layer_shape[-1]%2 == 0:
            sse_weights_dict['w1'] = tf.Variable(initializer(shape = [self.layer_shape[-1], int(self.layer_shape[-1]/2)]))
            sse_weights_dict['w2'] = tf.Variable(initializer(shape = [int(self.layer_shape[-1]/2), self.layer_shape[-1]]))
            sse_weights_dict['b1'] = tf.Variable(initializer(shape = [ int(self.layer_shape[-1]/2)]))
            sse_weights_dict['b2'] = tf.Variable(initializer(shape = [ self.layer_shape[-1]]))
    
        else :
            sse_weights_dict['w1'] = tf.Variable(initializer(shape = [self.layer_shape[-1], int((self.layer_shape[-1]-1)/2)]))
            sse_weights_dict['w2'] = tf.Variable(initializer(shape = [int((self.layer_shape[-1]-1)/2), self.layer_shape[-1]]))
            sse_weights_dict['b1'] = tf.Variable(initializer(shape = [int((self.layer_shape[-1]-1)/2)]))
            sse_weights_dict['b2'] = tf.Variable(initializer(shape = [ self.layer_shape[-1]]))
        
        self.sse_weights_dict = sse_weights_dict
        
        
    def cSE(self):
        self.cSE_weights_initializer()
        channel_squeeze = tf.nn.conv2d(self.layer, self.cse_weights_dict['1_by_1_conv'], strides = [1,1,1,1], padding = 'VALID')
        channel_squeeze = tf.nn.sigmoid(channel_squeeze)
        
        return tf.multiply(self.layer, channel_squeeze)
    
    
    def sSE(self):
        self.sSE_weights_initializer()
        spatial_squeeze = tf.reduce_mean(self.layer, axis = [1,2])
        spatial_squeeze = tf.reshape(spatial_squeeze, shape = [self.layer_shape[0], self.layer_shape[-1]])
        spatial_squeeze = tf.nn.bias_add(tf.matmul(spatial_squeeze, self.sse_weights_dict['w1']), self.sse_weights_dict['b1'])
        spatial_squeeze = tf.nn.relu(spatial_squeeze)
        spatial_squeeze = tf.nn.bias_add(tf.matmul(spatial_squeeze, self.sse_weights_dict['w2']), self.sse_weights_dict['b2'])
        spatial_squeeze = tf.nn.sigmoid(spatial_squeeze)
        spatial_squeeze = tf.reshape(spatial_squeeze, [self.layer_shape[0], 1, 1, self.layer_shape[-1]])
        
        return tf.multiply(self.layer, spatial_squeeze)
        
    
    def _scSE_(self):
        channel_squeeze = self.cSE()
        spatial_squeeze = self.sSE()
        scse = channel_squeeze + spatial_squeeze
        
        return scse
        
        

