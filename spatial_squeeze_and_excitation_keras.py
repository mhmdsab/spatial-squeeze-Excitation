from keras.layers import Conv2D, Lambda, Dense, Multiply, Add
import keras.backend as K

class scSE:
    
    def __init__(self, layer):
        self.layer = layer
        self.layer_shape = K.int_shape(layer)
        
        
    def cSE(self):
        channel_squeeze = Conv2D(1, (1,1), kernel_initializer="he_normal", activation='sigmoid')(self.layer)
        channel_squeeze = Multiply()([self.layer, channel_squeeze])
        return channel_squeeze
    
    
    def sSE(self):

        spatial_squeeze = Lambda(lambda x :K.mean(x, axis = [1,2]))(self.layer)
        spatial_squeeze = Dense(self.layer_shape[-1]//2, activation='relu')(spatial_squeeze)
        spatial_squeeze = Dense(self.layer_shape[-1], activation='sigmoid')(spatial_squeeze)
        spatial_squeeze = Multiply()([spatial_squeeze, self.layer])
        return spatial_squeeze
        
    
    def _scSE_(self):
        channel_squeeze = self.cSE()
        spatial_squeeze = self.sSE()
        scse = Add()([channel_squeeze, spatial_squeeze])
        
        return scse
        
        

