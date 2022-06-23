import tensorflow.keras as keras
from tensorflow.keras import layers, losses, regularizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tools_io import save, load
import VocalTractLab as vtl
from tensorflow.keras.backend import one_hot
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from vocal_learning import Agent, Optimization_State



def custom_loss( fwm, ivm ):
	def combined_loss( y_true, y_pred ):
		#mse = MeanSquaredError()( y_true[ :, :, 19], y_pred[ :, :, 19 ] )
		mse = loss_1( y_true, y_pred, ivm )
		#print( 'y_true: ', y_true )
		#print( 'y_pred: ', y_pred )
		#print( 'model y_pred', model( y_pred )  )
		#mse = 0
		#mse = MeanSquaredError()( y_true, y_pred )
		cse = CategoricalCrossentropy()( y_true[ :, :, 19 : ], fwm( y_pred[ :, :, : 19 ] ) )
		#cse = 0
		return mse + cse
	return combined_loss

def loss_1( y_true, y_pred, ivm ):
	return MeanSquaredError()( ivm( y_true[ :, :,  19: ] ), y_pred[ :, :, : 19 ] )


class Custom_MSE_CSE( keras.losses.Loss ):
	def __init__( self, fwm, ivm, name="custom_mse_cse" ):
		super().__init__(name=name)
		self.fwm = fwm
		self.ivm = ivm
		return

	def call( self, y_true, y_pred ):
		return self._mse( y_true, y_pred ) + self._cse_2( y_true, y_pred ) + self._cse( y_true, y_pred ) #+ self._cse_3( y_true, y_pred ) 

	def _mse( self, y_true, y_pred ):
		#return MeanSquaredError()( self.ivm( y_true[ :, :,  19: ] ), y_pred[ :, :, : 19 ] )
		return MeanSquaredError()( self.ivm( y_true[ :, :,  19: ] ), y_true[ :, :,  : 19 ] )
		#return MeanSquaredError()( self.ivm( self.fwm( y_true[ :, :, : 19 ] ) ), y_true[ :, :,  : 19 ] )

	def _cse( self, y_true, y_pred ):
		#return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )
		return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_true[ :, :, : 19 ] ) )

	def _cse_2( self, y_true, y_pred ):
		#return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )
		return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )

	def _cse_3( self, y_true, y_pred ):
		#return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )
		return CategoricalCrossentropy()( self.fwm( y_true[ :, :, : 19 ] ), self.fwm( y_pred[ :, :, : 19 ] ) )





class Custom_YM_Loss( keras.losses.Loss ):
	def __init__( self, fwm, ivm, name="custom_ym_mse_cse" ):
		super().__init__(name=name)
		self.fwm = fwm
		return
	def call( self, y_true, y_pred ):
		return self._mse( y_true, y_pred ) + self._cse( y_true, y_pred )
	def _cse( self, y_true, y_pred ):
		#return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )
		return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_true[ :, :, : 19 ] ) )
	def _mse( self, y_true, y_pred ):
		return MeanSquaredError()( y_true[ :, :,  : 19 ], y_pred[ :, :, : 19 ] )







class Custom_FWM_Loss( keras.losses.Loss ):
	def __init__( self, fwm, name="custom_mse_cse" ):
		super().__init__(name=name)
		self.fwm = fwm
		return

	def call( self, y_true, y_pred ):
		return MeanSquaredError()( y_true[ :, :, : 19 ], y_pred[ :, :, : 19 ] ) + self._cse( y_true, y_pred )

	def _cse( self, y_true, y_pred ):
		#return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )
		return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )





class Custom_IVM_Loss( keras.losses.Loss ):
	def __init__( self, fwm, name="custom_mse_cse" ):
		super().__init__(name=name)
		self.fwm = fwm
		return

	def call( self, y_true, y_pred ):
		return MeanSquaredError()( y_true[ :, :,  : 19 ], y_pred[ :, :, : 19 ] ) + self._cse( y_true, y_pred )

	def _cse( self, y_true, y_pred ):
		#return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )
		return CategoricalCrossentropy()( y_true[ :, :, 19 : ], self.fwm( y_pred[ :, :, : 19 ] ) )

	def _mse( self, y_true, y_pred ):
		return MeanSquaredError()( y_true[ :, :,  : 19 ], y_pred[ :, :, : 19 ] )




def custom_loss_2( model ):
	def combined_loss( y_true, y_pred ):
		mse = MeanSquaredError()( y_true[ :, 0, : 19 ], y_pred[ :, 0, : 19 ] )
		cse = CategoricalCrossentropy()( y_true[ :, 0, 19: ], model( y_pred[ :, 0, : 19 ] ) )
		return mse + cse
	return combined_loss



def forward_model( articulatory_dim = 19, n_classes = 37, compile_model = False ):
	inputs = keras.Input( shape=( 1, articulatory_dim ) )
	x = inputs
	x = layers.BatchNormalization()( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	outputs = layers.Dense( n_classes, activation = 'softmax' )( x )
	model = keras.Model( inputs, outputs , name = 'Forward_Model' )
	#print( model.summary() )
	if compile_model:
		model.compile(
			loss="categorical_crossentropy",
			optimizer=keras.optimizers.Adam(learning_rate=1e-4),
			metrics=["categorical_accuracy"],
		)
	print( model.summary() )
	return model

def inverse_model( n_classes = 37, articulatory_dim = 19, compile_model = False ):
	inputs = keras.Input( shape=( 1, n_classes ) )
	x = inputs
	#x = layers.BatchNormalization()( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	outputs = layers.Dense( articulatory_dim, activation = 'linear' )( x )
	model = keras.Model( inputs, outputs , name = 'Inverse_Model' )
	#print( model.summary() )
	if compile_model:
		model.compile(
			loss="mse",
			optimizer=keras.optimizers.Adam(learning_rate=1e-4),
			metrics=["mse"],
		)
	print( model.summary() )
	return model

def inverse_model_regularized( fwm, n_classes = 37, articulatory_dim = 19, compile_model = False ):
	#inputs2 = keras.Input( shape=( 1, articulatory_dim ) )
	inputs = keras.Input( shape=( 1, n_classes ) )
	x = inputs
	#x = layers.BatchNormalization()( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	x = layers.Dense( 32, activation= 'relu' )( x )
	outputs_temp = layers.Dense( articulatory_dim, activation = 'linear' )( x )
	outputs = keras.layers.concatenate( [ outputs_temp, inputs ] )
	model = keras.Model( inputs, outputs , name = 'Inverse_Model' )
	#print( model.summary() )
	cl = Custom_IVM_Loss( fwm )
	if compile_model:
		model.compile(
			loss=cl,
			optimizer=keras.optimizers.Adam(learning_rate=1e-4),
			metrics=["mse", cl._cse ],
		)
	print( model.summary() )
	return model


def forward_backward_model( model_1, model_2 ):
	#inputs_w = keras.Input( shape=( 1, 37 ) )
	inputs = keras.Input( shape=( 1, 19 ) )
	latent = model_1( inputs )
	outputs_temp = model_2( latent )
	outputs = keras.layers.concatenate( [ outputs_temp, latent ] )
	model = keras.Model( inputs, outputs , name = 'Forward_Inverse_Model' )
	#print( model.summary() )
	#if compile_model:
	#loss = custom_loss(model_1, inputs)
	#model.add_loss( loss )

	#cl = Custom_MSE_CSE( model_1, model_2 )
	#model.compile(
	#	#loss=custom_loss( model_1, inputs_w ),
	#	#loss = custom_loss(model_1, model_2 ),
	#	loss = cl,
	#	optimizer=keras.optimizers.Adam(learning_rate=1e-4),
	#	metrics = [ cl._mse, cl._cse, cl._cse_2 ]
	#	#metrics=["mse"],
	#)
	cl = Custom_YM_Loss( model_1, model_2 )
	model.compile(
		#loss=custom_loss( model_1, inputs_w ),
		#loss = custom_loss(model_1, model_2 ),
		loss = cl,
		optimizer=keras.optimizers.Adam(learning_rate=1e-4),
		metrics = [ cl._mse, cl._cse ]
		#metrics=["mse"],
	)
	print( model.summary() )
	return model

#def forward_backward_forward_model( model_1, model_2 ):
#	inputs = keras.Input( shape=( 1, 19 ) )
#	x = model_1( inputs )
#	x= model_2( x )
#	outputs_temp = model_1( x )
#	outputs = keras.layers.concatenate([inputs, outputs_temp])
#	model = keras.Model( inputs, outputs , name = 'Forward_Inverse_Model' )
#	#print( model.summary() )
#	#if compile_model:
#	model.compile(
#		loss="categorical_crossentropy",
#		optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#		metrics=["categorical_accuracy"],
#	)
#	print( model.summary() )
#	return model

def backward_forward_model( model_1, model_2 ):
	inputs = keras.Input( shape=( 1, 8 ) )
	x = model_1( inputs )
	outputs = model_2( x )
	model = keras.Model( inputs, outputs , name = 'Forward_Inverse_Model' )
	#print( model.summary() )
	#if compile_model:
	model.compile(
		loss="categorical_crossentropy",
		optimizer=keras.optimizers.Adam(learning_rate=1e-4),
		metrics=["categorical_accuracy"],
	)
	print( model.summary() )
	return model


def dummy_loss(y_true, y_pred):
    return 0.

if __name__ == '__main__':
	#X = load( 'results/data/results_VOWELS_states_X.pkl.gzip' )
	#y = load( 'results_VOWELS_states_y.pkl.gzip' )
	#y_pred = load( 'results/data/results_VOWELS_states_y_pred.pkl.gzip' )
	#X = load( 'results/data/results_data_states_X.pkl.gzip' )
	#y_pred = load( 'results/data/results_data_states_y.pkl.gzip' )
	#X = load( 'results/data/results_data_VISUAL_states_X.pkl.gzip' )
	#y_pred = load( 'results/data/results_data_VISUAL_states_y.pkl.gzip' )

	X = np.concatenate( [ load( 'results/data/results_data_states_X.pkl.gzip' ), load( 'results/data/results_data_VISUAL_states_X.pkl.gzip' ) ] )
	y_pred = np.concatenate( [ load( 'results/data/results_data_states_y.pkl.gzip' ), load( 'results/data/results_data_VISUAL_states_y.pkl.gzip' ) ] )
	print( X.shape )
	print( y_pred.shape )
	X_train, X_test, y_train, y_test = train_test_split( X, y_pred, train_size = 0.9, random_state=42, shuffle = True, stratify = [ np.argmax( y ) for y in y_pred ] )
	X_train = np.reshape( X_train, ( X_train.shape[0], 1, X_train.shape[1] ) )
	y_train = np.reshape( y_train, ( y_train.shape[0], 1, y_train.shape[1] ) )
	print( y_train.shape )
	#fb_y = np.concatenate( [ X_train[ 200000 : 400000 ], y_train[ 200000 : 400000 ] ], axis = 2 )
	fb_y = np.concatenate( [ X_train, y_train ], axis = 2 )
	#fb_y = np.concatenate( [ X_train[ : 200000 ], y_train[ : 200000 ] ], axis = 2 )
	print( fb_y.shape )
	#stop
	ivm = inverse_model( compile_model = True )
	fwm = forward_model( compile_model = True )
	#ivm.fit( y_train[ : 200000 ], X_train[ : 200000 ], validation_split = 0.1,  batch_size = 128, epochs = 10 )
	ivm.fit( y_train, X_train, validation_split = 0.1,  batch_size = 128, epochs = 100 )


	ivm.save( 'IVM_FULL_100_epochs_COMB.h5' )
	stop

	#for index, phoneme in enumerate( [ 'a', 'e', 'i', 'o', 'u', 'capE', '2', 'y' ] ):
	#	category = np.array( one_hot( [ index ], 37 ) ).reshape( (1,1,37) )
	#	category += 1/36*0.1
	#	category[0,0,index] = 0.9
	#	agent.synthesize( vtl.Supra_Glottal_Sequence( ivm.predict( category )[0] ), glt, 'results/VOWELS/distal_ivm_fwm/ivm_test_{}.wav'.format( phoneme ) )

	#fwm.fit( X_train, y_train, validation_split = 0.1, batch_size = 128, epochs = 100 )
	#ivm = inverse_model_regularized( fwm, compile_model = True )
	#ivm.fit( y_train, fb_y, validation_split = 0.1,  batch_size = 128, epochs = 20 )
	#ivm.save( 'IVM_REG_VOWELS.h5' )
	#agent = load( 'results/VOWELS_VISUAL/woa/u/0/agent.pkl.gzip' )
	#glt = vtl.get_shapes( 'modal' )
	#ivm = keras.models.load_model( 'IVM_REG_VOWELS.h5', custom_objects = { 'Custom_IVM_Loss' : Custom_IVM_Loss(fwm) }, compile = False )
	#for index in range(0, 15):
	#	category = np.array( one_hot( [ index ], 37 ) ).reshape( (1,1,37) )
	#	#category += 1/36*0.1
	#	#category[0,0,index] = 0.9
	#	agent.synthesize( vtl.Supra_Glottal_Sequence( ivm.predict( category )[0][ :, : 19 ] ), glt, 'results/VOWELS/distal_results/transfer_ivm_reg_run_1_{}.wav'.format( index ) )
	#stop

	#fwm.save( 'FWM_FULL.h5' )

	#stop

	#model = forward_backward_forward_model( fwm, ivm )
	#model.fit( X_train[ : 100000 ], y_train[ : 100000 ], validation_split = 0.1, batch_size = 128, epochs = 50 )
	#for index, phoneme in enumerate( [ 'a', 'e', 'i', 'o', 'u', 'capE', '2', 'y' ] ):
	#	category = np.array( one_hot( [ index ], 8 ) ).reshape( (1,1,8) )
	#	agent.synthesize( vtl.Supra_Glottal_Sequence( ivm.predict( category )[0] ), glt, 'results/VOWELS/distal_ivm_fwm/fwd_ivm_test_{}.wav'.format( phoneme ) )
	#stop

	#fb_y = np.array( X_train[ 200000 : 300000 ], y_train[ 200000 : 300000 ] )
	#print( fb_y.shape )

	ivm = keras.models.load_model( 'IVM_FULL.h5' )
	fwm = keras.models.load_model( 'FWM_FULL.h5' )

	model = forward_backward_model( fwm, ivm )
	#model.fit( X_train[ 200000 : 400000 ], fb_y, validation_split = 0.1, batch_size = 128, epochs = 50 )
	model.fit( X_train, fb_y, validation_split = 0.1, batch_size = 128, epochs = 10 )
	#model.fit( X_train[ : 200000 ], fb_y, validation_split = 0.1, batch_size = 128, epochs = 50 )
	#stop

	ivm.save( 'YM_IVM_CONCAT_FULL.h5' )
	fwm.save( 'YM_FWM_CONCAT_FULL.h5' )

	stop

	#model = backward_forward_model( ivm, fwm )
	#model.fit( y_train[ 100000 : 200000 ], y_train[ 100000 : 200000 ],  validation_split = 0.1, batch_size = 128, epochs = 50 )
	
	agent = load( 'results/VOWELS_VISUAL/woa/u/0/agent.pkl.gzip' )
	glt = vtl.get_shapes( 'modal' )
	ivm = keras.models.load_model( 'IVM_VOWELS.h5' )
	for index in range(0, 15):
		category = np.array( one_hot( [ index ], 37 ) ).reshape( (1,1,37) )
		#category += 1/36*0.1
		#category[0,0,index] = 0.9
		agent.synthesize( vtl.Supra_Glottal_Sequence( ivm.predict( category )[0] ), glt, 'results/audio/distal_results/transfer_ivm_run_1_{}.wav'.format( index ) )
	ivm = keras.models.load_model( 'YM_IVM_CONCAT_VOWELS.h5' )
	for index in range(0, 15):
		category = np.array( one_hot( [ index ], 37 ) ).reshape( (1,1,37) )
		#category += 1/36*0.1
		#category[0,0,index] = 0.9
		agent.synthesize( vtl.Supra_Glottal_Sequence( ivm.predict( category )[0] ), glt, 'results/audio/distal_results/transfer_ivm_concat_run_1_{}.wav'.format( index ) )

