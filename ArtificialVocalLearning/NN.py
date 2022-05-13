import tensorflow.keras as keras
from tensorflow.keras import layers, losses, regularizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import tensorflow as tf
#import tensorflow.keras.backend as K
#from tensorflow_addons.metrics import F1Score
#import tensorflow.keras.backend as K
#from tensorflow.keras.losses import SparseCategoricalCrossentropy
#from tensorflow.keras.losses import categorical_crossentropy




def GRU_Model_Large( time_dim, freq_dim, n_classes, compile_model = False ):
	inputs = keras.Input( shape=( time_dim, freq_dim ) )
	x = inputs
	x = layers.Masking( mask_value=[ 0 ], input_shape=( time_dim, freq_dim ) )( inputs )
	x = layers.BatchNormalization()( x )
	x = layers.Bidirectional( layers.GRU( 256, activation= 'tanh', return_sequences=True ) )( x )
	x = layers.Bidirectional( layers.GRU( 256, activation= 'tanh', return_sequences=True ) )( x )
	x = layers.Bidirectional( layers.GRU( 256, activation= 'tanh', return_sequences=True ) )( x )
	x = layers.Bidirectional( layers.GRU( 256, activation= 'tanh', return_sequences=True ) )( x )
	x = layers.Bidirectional( layers.GRU( 256, activation= 'tanh', return_sequences=False ) )( x )
	outputs = layers.Dense( n_classes, activation = 'softmax' )( x )
	model = keras.Model( inputs, outputs , name = 'GRU_Model' )
	#print( model.summary() )
	if compile_model:
		model.compile(
			loss="sparse_categorical_crossentropy",
			optimizer=keras.optimizers.Adam(learning_rate=1e-4),
			metrics=["sparse_categorical_accuracy"],
		)
	#print( model.summary() )
	return model