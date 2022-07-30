

#import base64
#import io
#from flask import request
#from flask import jsonify
#from flask import Flask
import os
import VocalTractLab as vtl
import numpy as np
import librosa
from copy import deepcopy
import pyloudnorm as pyln
from itertools import chain
from ArtificialVocalLearning.NN import GRU_Model_Large
from ArtificialVocalLearning.phoneme_classes import classes
from ArtificialVocalLearning.phoneme_classes import sequence_encoder, sequence_decoder

#app = Flask( __name__ )

feature_dimension = [ 125, 80 ]

#---------------------------------------------------------------------------------------------------------------------------------------------------#
def fade_in_fade_out( data_in, fade_to_sample, fade_from_sample ):
	data_faded_in = fade_in(
		data_in = data_in,
		fade_to_sample = fade_to_sample,
		)
	data_faded_in_out = fade_out(
		data_in= data_faded_in,
		fade_from_sample = fade_from_sample,
		)
	return data_faded_in_out
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def fade_in( data_in, fade_to_sample ):
	data_out = deepcopy( data_in )
	data_out[ : fade_to_sample ] = [
		x * cosine_window(
			x = index,
			a = len( data_in[ : fade_to_sample ] ),
			b = np.pi,
			) 
		for index, x in enumerate(
			data_in[ : fade_to_sample ]
			)
		]
	return data_out
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def fade_out( data_in, fade_from_sample ):
	data_out = deepcopy( data_in )
	data_out[ fade_from_sample : ] = [
		x * cosine_window(
			x = index,
			a = len( data_in[ fade_from_sample : ] )
			)
		for index, x in enumerate(
			data_in[ fade_from_sample : ] )
		]
	return data_out
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def cosine_window( x, a = 1, b = 0 ):
	return  0.5 * np.cos( np.pi * x / a + b ) + 0.5
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def make_audio( audio, orig_sr , onset = None, offset = None):
	resampled_audio = librosa.resample(
		y = audio,
		orig_sr = orig_sr,
		target_sr = 16000,
		)
	res_audio = np.concatenate(
			[ onset, 
			fade_in_fade_out( resampled_audio, 16, len( resampled_audio ) - 16 ), 
			offset ] )
	peak_normalized_audio = pyln.normalize.peak( res_audio, -1.0 )
	return peak_normalized_audio
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def preprocess( audio_in ):
	audio = make_audio(
		audio = audio_in,
		onset = np.zeros( int( 0.016 * 16000 ) ),
		offset =  np.zeros( int( 0.016 * 16000 ) ),
		orig_sr = 16000,
		)
	spectrogram = np.abs(
		librosa.stft(
			y = audio,
			**vtl.audio_tools.standard_16kHz_spectrogram_kwargs,
			)
		)**2
	melspectrogram = librosa.feature.melspectrogram(
		S = spectrogram,
		**vtl.audio_tools.standard_16kHz_melspectrogram_80_kwargs,
		)
	melspectrogram = librosa.power_to_db( melspectrogram )
	X = np.zeros( ( 1, *feature_dimension ) )
	X_feature = melspectrogram.T
	if feature_dimension[0] < X_feature.shape[0]:
		limit = feature_dimension[0]
	else:
		limit = X_feature.shape[0]
		X[ 0, : limit, : X_feature.shape[1] ] = X_feature[ : limit, :]
	return X
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def single_phoneme_recognition_model( phoneme_model_save_dir = 'models/RUN_2_tr_KIEL_BITS_te_VTL' ):
	phoneme_recognition_model = GRU_Model_Large(
		time_dim = feature_dimension[0],
		freq_dim = feature_dimension[1],
		n_classes = len( classes ),
		compile_model = True,
		)
	checkpoint_filepath = os.path.join( os.path.dirname( __file__ ), phoneme_model_save_dir, 'weights_best', 'checkpoint' )
	phoneme_recognition_model.load_weights( checkpoint_filepath )
	return phoneme_recognition_model
#---------------------------------------------------------------------------------------------------------------------------------------------------#

#def get_model():
#	phoneme_model_save_dir = 'models/RUN_2_tr_KIEL_BITS_te_VTL'
#	global model
#	model = single_phoneme_recognition_model( phoneme_model_save_dir )
#	print( ' * Model loaded!' )
#	return


print( ' * Loading Keras model...' )
#get_model()

#@app.route( '/predict', methods = [ 'POST' ] )
#def predict():
#	#message = request.get_json( force = True )
#	data = request.json
#	#encoded = message[ 'audio' ]
#	#decoded = base64.b64decode( encoded )
#	audio = np.frombuffer( base64.b64decode( data[ 'audio' ] ), dtype=float )
#	#audio = np.fromstring( data[ 'audio' ], dtype=float )
#	X = preprocess( audio )
#	prediction = model.predict( X )[ 0 ]
#	response = dict(
#		prediction = prediction.tolist()
#		)
#	return jsonify( response )
