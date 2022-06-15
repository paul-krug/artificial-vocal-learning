import VocalTractLab as vtl
import numpy as np
import pandas as pd
import os
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from scipy.spatial.distance import correlation
import umap
from tools_io import save, load

from sklearn.decomposition import PCA

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import one_hot
#from tensorflow.keras.utils import to_categorical
import random

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

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import fbeta_score, make_scorer
from tensorflow.keras.metrics import CategoricalAccuracy

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import time

from kneed import DataGenerator, KneeLocator

#vowels = [ 'a', 'e', 'i', 'o', 'u', 'E', 'I', 'O', 'U', '2', '9', 'y', 'Y', '@', '6' ]
#consonants = vtl.single_consonants()
#consonants.remove( 'T' )
#consonants.remove( 'D' )
#consonants.remove( 'r' )
#classes = [ x for x in chain( vowels, consonants ) ]
##classes.append( '<empty>' )
#classes = np.array( classes )
#print( classes )
#print( 'Nr of classes: ', classes.shape )
##---------------------------------------------------------------------------------------------------------------------------------------------------#
#def sequence_encoder( phoneme_sequence ):
#	labels = []
#	for phoneme in phoneme_sequence:
#		labels.append( np.where( classes == phoneme )[0][0] )
#	return np.array( labels )
##---------------------------------------------------------------------------------------------------------------------------------------------------#
#def sequence_decoder( label_sequence ):
#	phonemes = []
#	for label in label_sequence:
#		phonemes.append( classes[ label ] )
#	return np.array( phonemes )
##---------------------------------------------------------------------------------------------------------------------------------------------------#

def forward_model( articulatory_dim = 19, n_classes = 37, compile_model = False ):
	inputs = keras.Input( shape=( articulatory_dim ) )
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

def random_forest_classification( X, y ):
	X_train, X_test, y_train, y_test = train_test_split( X, y, train_size = 0.9, stratify=y, random_state=42, shuffle = True )
	#A random forest classifier will be fitted to compute the feature importances.
	from sklearn.ensemble import RandomForestClassifier
	#feature_names = [f"feature {i}" for i in range(X.shape[1])]
	feature_names = vtl.get_shapes( 'a' ).states.columns
	forest = RandomForestClassifier(random_state=0)
	forest.fit(X_train[ : 10000 ], y_train[ : 10000 ] )
	print( 'Forest score: {}'.format( forest.score( X_test, y_test ) ) )
	#stop
	import time
	import numpy as np
	start_time = time.time()
	importances = forest.feature_importances_
	print( importances )
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	elapsed_time = time.time() - start_time
	print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
	#Out:
	#Elapsed time to compute the importances: 0.008 seconds
	#Let’s plot the impurity-based importance.
	#import pandas as pd
	forest_importances = pd.Series(importances, index=feature_names)
	fig, ax = plt.subplots()
	forest_importances.plot.bar(yerr=std, ax=ax)
	ax.set_title("Feature importances using MDI")
	ax.set_ylabel("Mean decrease in impurity")
	fig.tight_layout()
	plt.show()
	#stop
	from sklearn.inspection import permutation_importance
	start_time = time.time()
	result = permutation_importance(
	    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
	)
	elapsed_time = time.time() - start_time
	print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
	forest_importances = pd.Series(result.importances_mean, index=feature_names)
	fig, ax = plt.subplots()
	forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
	ax.set_title("Feature importances using permutation on full model")
	ax.set_ylabel("Mean accuracy decrease")
	fig.tight_layout()
	plt.show()


ph_category_data = []
category_data = []
y_pre = []
#agents = []

category_dic = {}
category_dic_loss = {}

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(handle_unknown='ignore')
#enc.fit( np.array( [ [ 0, 1, 2 ] ] ).T )

#
#for c_index, consonant in enumerate( [ 'b', 'd', 'g' ] ):
#	grouped = []
#	for v_index, vowel in enumerate( [ 'a', 'i', 'u' ] ):
#		for x in range( 0, 10 ):
#			log_category_data = vtl.load( 'results/CONSONANTS/woa/{}_{}/{}/log_category_data.pkl.gzip'.format( consonant, vowel, x ) )
#			grouped.extend( log_category_data )
#	X = np.array( [ x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy() for x in grouped ] )
#	p = np.array( [ x[ 'total_phoneme_loss' ] for x in grouped ] )
#	category_dic[ consonant ] = X
#	category_dic_loss[ consonant ] = p
#save( category_dic, 'results/data/results_CONSONANTS_states_X_category_dic.pkl.gzip' )
#save( category_dic_loss, 'results/data/results_CONSONANTS_loss_p_category_dic.pkl.gzip' )
#category_dic = load( 'results/data/results_CONSONANTS_states_X_category_dic.pkl.gzip' )
#category_dic_loss = load( 'results/data/results_CONSONANTS_loss_p_category_dic.pkl.gzip' )
#for c_index, consonant in enumerate( [ 'b', 'd', 'g' ] ):
#	df = pd.DataFrame( [ [proba, state] for proba, state in zip( category_dic_loss[ consonant ], category_dic[ consonant] ) ], columns = [ 'y_pred', 'state' ] )
#	df_sorted = df.sort_values( 'y_pred' )
#	#print( df_sorted )
#	from vocal_learning import Agent, Optimization_State
#	agent = load( 'results/CONSONANTS/woa/b_a/0/agent.pkl.gzip' )
#	sglt = vtl.get_shapes( [ '@', '@' ] )
#	glt = vtl.get_shapes( [ 'modal', 'modal' ] )
#	#print( 'Vowel {}, lowest state: {}'.format( index, list( df_sorted[ 'state' ].iloc[ -1 ] ) ) )
#	for best in range( 1, 11 ):
#		sglt.states.iloc[0] = df_sorted[ 'state' ].iloc[ -1*best ]
#		agent.synthesize( 
#			sglt,
#			glt,
#			'results/audio/CONSONANTS_{}_{}.wav'.format( consonant, best )
#			)
#stop
#category_dic = load( 'results/data/results_CONSONANTS_states_X_category_dic.pkl.gzip' )
#y=[]
#x_sub=[]
#for idx, phoneme in enumerate( [ 'b', 'd', 'g' ] ):
#	x_sub.append( category_dic[ phoneme ] )
#	y.extend( [ idx for _ in range( 0, len( category_dic[ phoneme ] ) ) ] )
#X = np.concatenate( list( category_dic.values() ) )
#print( X.shape )
#X_embedded = umap.UMAP( n_neighbors = 100, min_dist = 0.001, init='random', random_state = 12345 ).fit_transform( X )
#plt.scatter( X_embedded[ :, 0 ], X_embedded[:, 1], c = y )
#plt.colorbar()
#plt.show()
#stop


#grouped = []

#for index, phoneme in enumerate( [ 'a', 'e', 'i', 'o', 'u', 'capE', '2', 'y' ] ):
#	category = np.array( one_hot( [ index ], 8 ) ).reshape( (8) )
#	print(category)
#	#ph
#	grouped_group = []
#	for g in range( 0, 10 ):
#		grouped = []
#		for x in range( 0, 10 ):
#			#category_data.extend( vtl.load( 'demo_run/CONSONANT_VTL_PRESET_VOWELS/woa/d_i/{}/log_category_data.pkl.gzip'.format( x ) ) )
#			#log_category_data = vtl.load( 'results/VOWELS/woa/{}/{}/log_category_data.pkl.gzip'.format( phoneme, x ) )
#			log_category_data = vtl.load( 'results/VOWELS_VISUAL/woa/{}/{}/log_category_data.pkl.gzip'.format( phoneme, g * 10 + x ) )
#			#for _ in range( 0, len( log_category_data ) ):
#			#	y_pre.append( category )
#			#category_data.extend( log_category_data )
#			grouped.extend( log_category_data )
#		X = np.array( [ x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy() for x in grouped ] )
#		#p = np.array( [ x[ 'total_phoneme_loss' ] for x in grouped ] )
#		grouped_group.append( X )
#		#grouped_group.append( p )
#	category_dic[ phoneme ] = grouped_group
##save( category_dic, 'results/data/results_VOWELS_VISUAL_loss_p_category_dic.pkl.gzip' )
#save( category_dic, 'results/data/results_VOWELS_VISUAL_states_X_category_dic.pkl.gzip' )
#stop

#for index, phoneme in enumerate( [ 'a', 'e', 'i', 'o', 'u', 'capE', '2', 'y' ] ):
#	category = np.array( one_hot( [ index ], 8 ) ).reshape( (8) )
#	print(category)
#	for x in range( 0, 100 ):
#		log_category_data = vtl.load( 'results/VOWELS_VISUAL/woa/{}/{}/log_category_data.pkl.gzip'.format( phoneme, x ) )
#		category_data.extend( log_category_data )
#loss = np.array( [ x[ 'total_loss' ] for x in category_data ] )
#save( loss, 'results_VOWELS_VISUAL_total_loss.pkl.gzip' )
#stop

#y_pred = np.array( [ list( x[ 'phoneme_recognition_states' ].values() )[0] for x in category_data ] )
#print( y_pred.shape )
#save( y_pred, 'results_VOWELS_VISUAL_states_y_pred.pkl.gzip' )
#X = np.array( [ x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy() for x in category_data ] )
#y = np.array( y_pre )
#save( X, 'results_VOWELS_VISUAL_states_X.pkl.gzip' )
#save( y, 'results_VOWELS_VISUAL_states_y.pkl.gzip' )
#stop

#save( X, 'results_VOWELS_states_X.pkl.gzip' )
#save( y, 'results_VOWELS_states_y.pkl.gzip' )

def my_custom_loss_func(y_true, y_pred):
	#return CategoricalAccuracy()( y_true, y_pred )
	return f1_score( np.argmax( y_true, axis = 1 ), np.argmax( y_pred, axis = 1), average = 'macro', zero_division = 0 )


def train_nn_forward_classifier( X, y, name ):
	permutation_1 = np.random.permutation( X.shape[0] )
	X = X[ permutation_1 ]
	y = y[ permutation_1 ]
	y_labels= np.array( [ np.argmax( val ) for val in y ] )
	skf = StratifiedKFold( n_splits = 10 )#, random_state=42, shuffle=True )
	score = make_scorer(my_custom_loss_func, greater_is_better=True)
	feature_names = vtl.get_shapes( 'a' ).states.columns
	for index, (train_index, test_index) in enumerate( skf.split( X, y_labels ) ):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		fwm = forward_model( n_classes = 8, compile_model = True )
		#X_train, X_test, y_train, y_test = train_test_split( X, y, train_size = 0.5, stratify=y, random_state=42, shuffle = True )
		#X_train = np.random.permutation( X_train, axis = 0 )
		permutation = np.random.permutation(X_train.shape[0])
		X_train = X_train[ permutation ]
		y_train = y_train[ permutation ]
		#y_train = np.random.permutation( y_train, axis = 0 )
		#np.take( y_train, np.random.permutation(y_train.shape[0]),axis=0, out= y_train )
		print( X_train.shape )
		print( X_test.shape )
		H = fwm.fit(
			X_train,#.reshape( X_train.shape[0], 1, X_train.shape[1] ),
			y_train,#.reshape( y_train.shape[0], 1, y_train.shape[1] ),
			validation_split = 0.1,
			batch_size = 128,
			epochs = 100,
			)
		save( H.history, 'results/{}_{}_history.pkl.gzip'.format( name, index ) )
		fwm.evaluate(
			X_test,#.reshape( X_test.shape[0], 1, X_test.shape[1] ),
			y_test,#.reshape( y_test.shape[0], 1, y_test.shape[1] ),
			batch_size = 128,
			)
		y_true = y_test
		y_pred = fwm.predict( X_test )
		f1 = f1_score( np.argmax( y_true, axis = 1 ), np.argmax( y_pred, axis = 1), average = 'macro', zero_division = 0 )
		precision = precision_score( np.argmax( y_true, axis = 1 ), np.argmax( y_pred, axis = 1), average = 'macro', zero_division = 0 )
		recall = recall_score( np.argmax( y_true, axis = 1 ), np.argmax( y_pred, axis = 1), average = 'macro', zero_division = 0 )
		metric_res = dict(
			f1 = f1,
			precision = precision,
			recall = recall,
			)
		save( metric_res, 'results/{}_{}_metric_res.pkl.gzip'.format( name, index ) )
		metric_res = load( 'results/{}_{}_metric_res.pkl.gzip'.format( name, index ) )
		#print( 'precision: {}, recall: {}, f1: {}'.format( precision, recall, f1 ) )
		print( metric_res )
		from sklearn.inspection import permutation_importance
		start_time = time.time()
		result = permutation_importance(
			fwm, X_test, y_test,
			n_repeats=10, random_state=42, n_jobs=1, scoring = score,
		)
		save( result, 'results/{}_{}_permuatation_result.pkl.gzip'.format( name, index ) )
		result = load( 'results/{}_{}_permuatation_result.pkl.gzip'.format( name, index ) )
		elapsed_time = time.time() - start_time
		print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
		#forest_importances = pd.Series(result.importances_mean, index=feature_names)
		#fig, ax = plt.subplots()
		#forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
		#ax.set_title("Feature importances using permutation on full model")
		#ax.set_ylabel("Mean accuracy decrease")
		#fig.tight_layout()
		#plt.show()
		#stop
	return

def plot_nn_forward_classifier():
	feature_names = vtl.get_shapes( 'a' ).states.columns
	df_imp = []
	df_list = []
	for name in [ 'forward_cross_validation', 'VISUAL_forward_cross_validation', 'VISUAL_COMB_forward_cross_validation' ]:
		df_list.append( pd.DataFrame( [ load( 'results/nn_forward_classifier/{}_{}_metric_res.pkl.gzip'.format( name, index ) ) for index in range( 0, 10 ) ] )[ 'f1' ] )
		results = [ load( 'results/nn_forward_classifier/{}_{}_permuatation_result.pkl.gzip'.format( name, index ) ) for index in range( 0, 10 ) ]
		df_importances = pd.DataFrame(
			np.array( [ np.mean( [ x.importances_mean for x in results ], axis = 0 ), np.std( [ x.importances_mean for x in results ], axis = 0 ) ] ).T,
			index = feature_names,
			columns = [ 'mean', 'std' ]
			)
		df_importances = df_importances.sort_values( 'mean', ascending = False )
		df_imp.append( df_importances )
		kneedle = KneeLocator([ x for x in range(0, len(df_importances.index) )], df_importances[ 'mean' ], S=1.0, curve="convex", direction="increasing", )# interp_method = 'polynomial', polynomial_degree = 4 )
		#kneedle.plot_knee_normalized()
		#plt.show()
		print(round(kneedle.knee, 3))
		#stop
		fig, axs = plt.subplots( figsize = ( 14, 4 ) )
		plt.bar( df_importances.index, df_importances[ 'mean' ] )
		#plt.xticks( [ x for x in range( 0, len( importances ) ) ], list( feature_names ) )
		plt.ylabel("Mean accuracy decrease")
		plt.tight_layout()
		plt.show()

	#plt.boxplot( df_list )
	#plt.show()

#from vocal_learning import Agent, Optimization_State
#agent = load( 'results/VOWELS/woa/u/0/agent.pkl.gzip' )
#glt = vtl.get_shapes( 'modal' )
#for preset in  [ 'a', 'e', 'i', 'o', 'u', 'E:', '2', 'y' ]:
#	sglt = vtl.get_shapes( preset )
#	if preset == 'E:':
#		preset = 'capE'
#	agent.synthesize( 
#		sglt,
#		glt,
#		'results/audio/VOWELS_VTL_PRESET_{}.wav'.format( preset )
#		)

#stop



X_data = load( 'results/data/results_VOWELS_VISUAL_states_X.pkl.gzip' )[ : ]
#y_data = load( 'results/data/results_VOWELS_states_y.pkl.gzip' )[ : ]
y_pred_data = load( 'results/data/results_VOWELS_VISUAL_states_y_pred.pkl.gzip' )[ : ]
loss_data = load( 'results/data/results_VOWELS_VISUAL_total_loss.pkl.gzip' )[ : ]



#for index in range( 0, 15 ):
for index in [ 0, 1, 2, 3, 4, 5, 9, 11 ]:
	#prob = [ x[ index ] for x in y_pred_data ]
	#print( np.array(prob).shape )
	df = pd.DataFrame( [ [proba, y_pred, state] for proba, y_pred, state in zip( loss_data, y_pred_data, X_data ) ], columns = [ 'loss', 'y_pred', 'state' ] )
	df[ 'y_pos' ] = [ np.argmax( x ) for x in df['y_pred'] ]
	df = df.loc[ df[ 'y_pos' ] == index ]
	df_sorted = df.sort_values( 'loss' )
	#print( df_sorted )
	from vocal_learning import Agent, Optimization_State
	agent = load( 'results/VOWELS/woa/u/0/agent.pkl.gzip' )
	glt = vtl.get_shapes( 'modal' )
	print( 'Vowel {}, lowest state: {}'.format( index, list( df_sorted[ 'state' ].iloc[ 0 ] ) ) )
	#for best in range( 0, 10 ):
	#	agent.synthesize( 
	#		vtl.Supra_Glottal_Sequence( df_sorted[ 'state' ].iloc[ best ].reshape( 1, 19 ) ),
	#		#vtl.Supra_Glottal_Sequence( df_sorted[ 'state' ].iloc[ -1*best ].reshape( 1, 19 ) ),
	#		glt,
	#		'results/audio/VISUAL_total_loss/VOWELS_VISUAL_{}_{}.wav'.format( index, best+1 )
	#		)


stop



#plot_nn_forward_classifier()
#stop

from vocal_learning import Agent, Optimization_State
agent = load( 'results/VOWELS_VISUAL/woa/u/0/agent.pkl.gzip' )
glt = vtl.get_shapes( 'modal' )


category_dic = load( 'results/data/results_VOWELS_VISUAL_states_X_category_dic.pkl.gzip' )
category_dic_loss = load( 'results/data/results_VOWELS_VISUAL_loss_p_category_dic.pkl.gzip' )


for index, phoneme in enumerate( [ 'u', 'a', 'e', 'i', 'o', 'u', 'capE', '2', 'y' ] ):
	error_selection = []
	event_selection = []
	#for run in range( 0, 5 ):
	#	#X = np.concatenate( category_dic[ phoneme ][:2] )
	#	X = category_dic[ phoneme ][ run ]#[ :, [ 1, 3, 4, 5, 8, 9 ] ]
	#	sgs = vtl.Supra_Glottal_Sequence( X )
	#	sgs.plot_distributions()
	#stop
	errors = []
	select_classes = []
	for run in range( 0, 5 ):
		X = category_dic[ phoneme ][ run ]#[ :, [ 1, 3, 4, 5, 8, 9 ] ]
		p = category_dic_loss[ phoneme ][ run ]
		#plt.hist(p, bins = 50 )
		#plt.show()
		#stop
		print( X.shape )
		X_embedded = umap.UMAP( n_neighbors = 100, min_dist = 0.001, init='random', random_state = 12345 ).fit_transform( X ) # articulatory vector embedding model
		clustering = DBSCAN( eps=1, min_samples=10 ).fit( X_embedded )
		#clustering = OPTICS(min_samples=50, metric='euclidean' ).fit(X_embedded)
		labels = clustering.labels_
		print( labels )
		#errors = []
		#select_classes = []
		for lbl_idx, cls_lbl in enumerate( set(labels) ):
			select_class = [ cls_idx for cls_idx,  label in enumerate( labels ) if label == cls_lbl ]
			errors.append( [ lbl_idx, run, np.median( p[ select_class ] ), select_class ] )
			#plt.hist( p[ select_class ], bins = 50, histtype = 'step' )
			#sgs = vtl.Supra_Glottal_Sequence( X[ select_class ] )
			#sgs.plot_distributions()
			select_classes.append( [ lbl_idx, run, select_class] )
		#error_selection.append( errors )
		#event_selection.append( select_classes )
	errors = pd.DataFrame( np.array( errors ), columns = [ 'lbl_idx', 'run', 'loss', 'select_class' ] )
	print( errors )
	print( errors.sort_values( 'loss' ) )
	errors = errors.sort_values( 'loss' )
	event_selection = errors[ 'select_class' ]
	run_selection = errors[ 'run' ]
	for best in range( 0, 10 ):
		#arr = np.median( X[ event_selection.iloc[ best ] ], axis = 0 )
		#print( arr.shape )
		agent.synthesize( 
			vtl.Supra_Glottal_Sequence( np.median( category_dic[ phoneme ][ run_selection.iloc[ best ] ][ event_selection.iloc[ best ] ], axis = 0 ).reshape( 1, 19 ) ),
			glt,
			'test_low_error_{}.wav'.format( best )
			)
	top
	error_selection = np.array( error_selection )
	print( error_selection )
	print( np.unravel_index(np.argsort(error_selection, axis=None), error_selection.shape) )
	stop
		#plt.show()
		#position = np.argmin( errors )
		#sgs = vtl.Supra_Glottal_Sequence( X[ select_classes[ position ] ] )
		#sgs.plot_distributions()
		#stop
		#plt.scatter( X_embedded[ :, 0 ], X_embedded[:, 1], c = p )
		#plt.colorbar()
		#plt.show()
		#stop
stop
stop







X_data = load( 'results/data/results_VOWELS_states_X.pkl.gzip' )[ : ]
y_data = load( 'results/data/results_VOWELS_states_y.pkl.gzip' )[ : ]



#train_nn_forward_classifier( X_data, y_data, 'forward_cross_validation' )
#stop

X_data_vis = load( 'results/data/results_VOWELS_VISUAL_states_X.pkl.gzip' )[ : ]
y_data_vis = load( 'results/data/results_VOWELS_VISUAL_states_y.pkl.gzip' )[ : ]

#train_nn_forward_classifier( X_data_vis, y_data_vis, 'VISUAL_forward_cross_validation' )
#stop

X = np.concatenate( [ X_data, X_data_vis ] )
y = np.concatenate( [ y_data, y_data_vis ] )

#plt.scatter( X_data_vis[ :, 4 ], X_data_vis[ :, 8], c = [ np.argmax(y_val) for y_val in y_data_vis ] )
#plt.show()
#stop
#fwm = forward_model( articulatory_dim = 5, n_classes = 8, compile_model = True )
#H = fwm.fit(
#	X[ :, [ 1, 3, 4, 5, 8 ] ],#.reshape( X_train.shape[0], 1, X_train.shape[1] ),
#	y,#.reshape( y_train.shape[0], 1, y_train.shape[1] ),
#	validation_split = 0.1,
#	batch_size = 128,
#	epochs = 10,
#	)
#stop
#train_nn_forward_classifier( X, y, 'VISUAL_COMB_forward_cross_validation' )
#stop
#train_nn_forward_classifier( X, y )
##random_forest_classification( X, y )
#stop

y_data = np.array( [ np.argmax( y ) for y in y_data ] )
y_data_vis = np.array( [ np.argmax( y ) for y in y_data_vis ] )


indices = []
for index in range( 0, 8 ):
	indices.append( [ x for x, y in enumerate( y_data ) if y == index ] )
	print( '{}: {}'.format( index, len(indices[-1]) ) )


indices_vis = []
for index in range( 0, 8 ):
	indices_vis.append( [ x for x, y in enumerate( y_data_vis ) if y == index ] )
	print( '{}: {}'.format( index, len(indices_vis[-1]) ) )


#fig, axs = plt.subplots( 2, 5, )

select = []
select_labels = []

select_vis = []
select_labels_vis = []
vtl_presets = [ 'a', 'e', 'i', 'o', 'u', 'E', '2', 'y' ]

preset_distances = []
preset_distances_vis = []
for index in range( 0, 8 ):
	###n_sample = 25000
	####select_from_category = random.sample( indices[index], n_sample )
	###select_from_category = indices[index]#[ : n_sample ]
	###select.extend( select_from_category )
	###select_labels.extend( [ index for _ in range( 0, len( select_from_category ) ) ] )
	####select_from_category_vis = random.sample( indices_vis[index], n_sample )
	###select_from_category_vis = indices_vis[index]#[ : n_sample ]
	###select_vis.extend( select_from_category_vis )
	###select_labels_vis.extend( [ index for _ in range( 0, len( select_from_category_vis ) ) ] )
	####avem = load( 'avem_category_{}_umap_nn_100.pkl.gzip'.format( index ) )
	####avem_dim_4 = load( 'avem_dim_4_category_{}_umap_nn_100.pkl.gzip'.format( index ) )
	###X = X_data[ select_from_category ]
	###X_vis = X_data_vis[ select_from_category_vis ]

	#sgs_visual = vtl.Supra_Glottal_Sequence( X )
	#sgs_visual.plot_distributions( out_file_path = 'results/VOWELS_category_{}_art_dist_25k.pdf'.format( index ) )
	#stop
	#X_embedded = avem.transform( X )
	#axs[0][ index ].scatter( X_embedded[ :, 0 ], X_embedded[:, 1] )
	#X = X[ :, [ 1, 4, 5, 8 ] ] # no visual

	#X = X[ :, [ 1, 3, 4, 5, 8, 9 ] ] # visual

	#X_embedded = avem_dim_4.transform( X )
	#axs[1][ index ].scatter( X_embedded[ :, 0 ], X_embedded[:, 1] )

	X_embedded = umap.UMAP( n_neighbors = 1000, min_dist = 0.001, init='random', random_state = 12345 ).fit_transform( X ) # articulatory vector embedding model
	plt.scatter( X_embedded[ :, 0 ], X_embedded[:, 1] )#c = labels )
	plt.colorbar()
	plt.show()
	stop
	vtl_state = vtl.get_shapes( vtl_presets[ index ] ).states.to_numpy()
	preset_distances.append(
		[ [ np.abs(
			vtl_state[ 0, param ] -
			X[ x, param ]
			) for x in range( 0, n_sample ) ]
			for param in [  1, 3, 4, 5, 8, 9 ]
			#for param in [  8, ]
		]
		)
	preset_distances_vis.append(
		[ [ np.abs(
			vtl_state[ 0, param ] -
			X_vis[ x, param ]
			) for x in range( 0, n_sample ) ]
			for param in [  1, 3, 4, 5, 8, 9 ]
			#for param in [  8, ]
		]
		)
	#for param in [  1, 3, 4, 5, 8, 9 ]:
	#	print(
	#		'MSE, param: {}: {}'.format(
	#			param,
	#			MeanSquaredError()(
	#				vtl.get_shapes( vtl_presets[ index ] ).states.to_numpy()[ :, param ],
	#				X[ :, param ],
	#				)
	#			)
	#		)
	#stop
	#save( avem, 'avem_dim_4_category_{}_umap_nn_100.pkl.gzip'.format( index ) )
	#print( 'finished training for category: ', index )
#plt.show()

from pylab import setp
def setBoxColors(bp, color_1 = 'navy', color_2 = 'darkmagenta' ):
    setp(bp['boxes'][0], color=color_1 )
    setp(bp['caps'][0],   color=color_1 )
    setp(bp['caps'][1],     color=color_1  )
    #setp(bp['whiskers'][0], color=color_1  )
    #setp(bp['whiskers'][1], color=color_1  )
    #setp(bp['fliers'][0], color= color_1 )
    #setp(bp['fliers'][1], color= color_1 )
    setp(bp['medians'][0], color=color_1 )

    setp(bp['boxes'][1], color=color_2)
    setp(bp['caps'][2], color= color_2)
    setp(bp['caps'][3], color= color_2)
    #setp(bp['whiskers'][2], color=color_2)
    #setp(bp['whiskers'][3], color=color_2)
    #setp(bp['fliers'][2], color=color_2)
    #setp(bp['fliers'][3], color=color_2)
    setp(bp['medians'][1], color=color_2)


preset_distances = np.array( preset_distances ).reshape( (6, 8*n_sample) )
preset_distances_vis = np.array( preset_distances_vis ).reshape( (6, 8*n_sample) )

fig, axs = plt.subplots( figsize = ( 14, 4 ) )

bp = plt.boxplot( [ preset_distances[0], preset_distances_vis[0] ], positions = [1, 2], widths = 0.6, notch = True, showfliers = False )
setBoxColors(bp)
bp = plt.boxplot( [ preset_distances[1], preset_distances_vis[1] ], positions = [4, 5], widths = 0.6, notch = True, showfliers = False )
setBoxColors(bp)
bp = plt.boxplot( [ preset_distances[2], preset_distances_vis[2] ], positions = [7, 8], widths = 0.6, notch = True, showfliers = False )
setBoxColors(bp)
bp = plt.boxplot( [ preset_distances[3], preset_distances_vis[3] ], positions = [10, 11], widths = 0.6, notch = True, showfliers = False )
setBoxColors(bp)
bp = plt.boxplot( [ preset_distances[4], preset_distances_vis[4] ], positions = [13, 14], widths = 0.6, notch = True, showfliers = False )
setBoxColors(bp)
bp = plt.boxplot( [ preset_distances[5], preset_distances_vis[5] ], positions = [16, 17], widths = 0.6, notch = True, showfliers = False )
setBoxColors(bp)
bp = plt.boxplot( [ preset_distances.flatten(), preset_distances_vis.flatten() ], positions = [19, 20], widths = 0.6, notch = True, showfliers = False )
setBoxColors(bp)

hB, = plt.plot([1,1],color = 'navy')
hR, = plt.plot([1,1],color = 'darkmagenta')
plt.legend((hB, hR),('Baseline', 'Visual'))
hB.set_visible(False)
hR.set_visible(False)

plt.xticks( [ 1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5 ], [ 'HY', 'JA', 'LP', 'LD', 'TCX', 'TCY', 'Total' ] )
#plt.hist( preset_distances[0], histtype = 'step' )
#plt.hist( preset_distances_vis[0], histtype = 'step' )
plt.ylabel( 'Absolute error' )
plt.tight_layout()
plt.show()

print( preset_distances.shape )

stop

#X = X_data[ select ]



print( y_data.shape )

unique, counts = np.unique(y_data, return_counts=True )
elements = dict(zip(unique, counts))

print( elements )
#stop

#y = load( 'results_VOWELS_states_y.pkl.gzip' )
print( X_data.shape )
#print( y.shape )
X = X_data[ select ]
#X = X[ :, [ 1, 4, 5, 8 ] ]
print( X.shape )


#X_embedded = umap.UMAP( n_neighbors = 4, min_dist = 0.0, init='random', random_state = 12345 ).fit_transform( X )
X_embedded = umap.UMAP( n_neighbors = 100, init='random', random_state = 12345 ).fit_transform( X )
#X_embedded = TSNE(n_components=2, perplexity = 100, learning_rate='auto',init='random', random_state = 12345, n_iter = 1000 ).fit_transform( X )
print( 'Embedded shape: ', X_embedded.shape )

plt.scatter( X_embedded[ :, 0 ], X_embedded[:, 1], c = select_labels )#c = labels )
plt.colorbar()
plt.show()

stop


X_embedded = umap.UMAP( n_neighbors = 4, min_dist = 0.0, init='random', random_state = 12345, n_epochs = 5000 ).fit_transform( X )
#X_embedded = TSNE(n_components=2, perplexity = 50, learning_rate='auto',init='random', random_state = 12345, n_iter = 5000 ).fit_transform( X )
print( 'Embedded shape: ', X_embedded.shape )
#stop
#XX = np.array( [ X_embedded[:,0], X_embedded[:,1], errors ] ).T
clustering = DBSCAN(eps=1, min_samples=50 ).fit(X_embedded)
#clustering = OPTICS(min_samples=50, metric='euclidean' ).fit(X_embedded)
labels = clustering.labels_
plt.scatter( X_embedded[ :, 0 ], X_embedded[:, 1] , c = labels )#, c = dist_error )#c = labels )
#for x, label in zip( X_embedded, labels ):
#    plt.text(x[0], x[1], str(label), color="red", fontsize=12)
plt.colorbar()
plt.show()

for label in set( labels ):
	#dist = np.array( [ x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy() for index, x in enumerate( category_data ) if labels[ index ] == label ] )
	dist = np.array( [ x for index, x in enumerate( X_data ) if labels[ index ] == label ] )
	#effort = np.array(
	#	[ correlation( x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy(), x[ 'supra_glottal_sequence' ].states.iloc[ 1, : ].to_numpy() )
	#		for index, x in enumerate( category_data ) if labels[ index ] == label ]
	#		)
	#dist_error = np.array( [ x[ 'total_loss' ] for index, x in enumerate( category_data ) if labels[ index ] == label ] )
	#dist_error_c = np.array( [ x[ 'phoneme_losses' ]['d'] for index, x in enumerate( category_data ) if labels[ index ] == label ] )
	#dist_error_v = np.array( [ x[ 'phoneme_losses' ]['i'] for index, x in enumerate( category_data ) if labels[ index ] == label ] )
	avg = np.reshape( np.median( dist, axis = 0 ), (1,19) )
	#synthesize_cx( avg, context = 'i', out_path = 'demo_run/avg_test_for_best_const/di/solution_{}.wav'.format(label) )
	#agent.synthesize( vtl.Supra_Glottal_Sequence( avg ), glt, 'results/VOWELS/avg_test_for_best/a/solution_{}.wav'.format(label) )
	#vtl.VocalTractLabApi._tract_sequence_to_svg(
	#	( vtl.Supra_Glottal_Sequence( avg ),
	#		'results/VOWELS/avg_test_for_best/a/c_solution_{}'.format(label),
	#		44100/110,
	#		)
	#	)
	sgs_dist = vtl.Supra_Glottal_Sequence( dist )
	sgs_dist.plot_distributions()
	#plt.hist( dist[ :, 3 ], label = label, histtype= 'step' )
	articulatory_distance = [ cosine( avg.reshape(19), x.reshape(19) ) for x in dist ]
	tsne_points = np.array( [ x for index, x in enumerate( X_embedded ) if labels[ index ] == label ] )
	tsne_distance = [ mean_squared_error( [np.mean(tsne_points[:,0]), np.mean(tsne_points[:,1] ) ], x ) for x in tsne_points ]
	print( '' )
	#print( 'Label: {}, median error: {}, mean_error: {}, std: {}'.format( label, np.median(dist_error), np.mean(dist_error), np.std(dist_error) ) )
	#print( 'Phone: C, median error: {}, mean_error: {}, std: {}'.format( np.median(dist_error_c), np.mean(dist_error_c), np.std(dist_error_c) ) )
	#print( 'Phone: V, median error: {}, mean_error: {}, std: {}'.format( np.median(dist_error_v), np.mean(dist_error_v), np.std(dist_error_v ) ) )
	print( 'Mean art. distance: {} +/- {}'.format( np.mean(articulatory_distance), np.std(articulatory_distance) ) )
	print( 'mean tsne distance: {} +/- {}'.format( np.mean(tsne_distance), np.std(tsne_distance) ) )
	#print( 'median effort: {} +/- {}'.format( np.median( effort), np.std( effort ) ) )
plt.show()

stop


#plt.matshow( pd.DataFrame(X).corr(), cmap='coolwarm' )
#plt.show()
#stop


stop
		
agent = load( 'results/VOWELS_VISUAL/woa/u/0/agent.pkl.gzip' )
glt = vtl.get_shapes( 'modal' )
print( len( category_data ) )
array = np.array( [ x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy() for x in category_data ])
errors = [ x[ 'total_loss' ] for x in category_data ]
print( array.shape )
sgs = vtl.Supra_Glottal_Sequence( array )
#sgs.plot_distributions()
#stop
#X_embedded = TSNE(n_components=2, perplexity = 50, learning_rate='auto',init='random', random_state = 12345 ).fit_transform(sgs.states.to_numpy() )
#X_PCA = PCA( n_components = 6 ).fit_transform( sgs.states.to_numpy() )
X_embedded = umap.UMAP( n_neighbors = 50, init='random', random_state = 12345 ).fit_transform( sgs.states.to_numpy() )
print( 'Embedded shape: ', X_embedded.shape )
#stop
XX = np.array( [ X_embedded[:,0], X_embedded[:,1], errors ] ).T
clustering = DBSCAN(eps=1, min_samples=10 ).fit(XX)
#clustering = OPTICS(min_samples=50, metric='euclidean' ).fit(X_embedded)
labels = clustering.labels_

#for label in set( labels ):
#	dist = np.array( [ x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy() for index, x in enumerate( category_data ) if labels[ index ] == label ] )
#	#effort = np.array(
#	#	[ correlation( x[ 'supra_glottal_sequence' ].states.iloc[ 0, : ].to_numpy(), x[ 'supra_glottal_sequence' ].states.iloc[ 1, : ].to_numpy() )
#	#		for index, x in enumerate( category_data ) if labels[ index ] == label ]
#	#		)
#	dist_error = np.array( [ x[ 'total_loss' ] for index, x in enumerate( category_data ) if labels[ index ] == label ] )
#	#dist_error_c = np.array( [ x[ 'phoneme_losses' ]['d'] for index, x in enumerate( category_data ) if labels[ index ] == label ] )
#	#dist_error_v = np.array( [ x[ 'phoneme_losses' ]['i'] for index, x in enumerate( category_data ) if labels[ index ] == label ] )
#	avg = np.reshape( np.median( dist, axis = 0 ), (1,19) )
#	#synthesize_cx( avg, context = 'i', out_path = 'demo_run/avg_test_for_best_const/di/solution_{}.wav'.format(label) )
#	agent.synthesize( vtl.Supra_Glottal_Sequence( avg ), glt, 'results/VOWELS/avg_test_for_best/u/solution_{}.wav'.format(label) )
#	vtl.VocalTractLabApi._tract_sequence_to_svg(
#		( vtl.Supra_Glottal_Sequence( avg ),
#			'results/VOWELS/avg_test_for_best/u/c_solution_{}'.format(label),
#			44100/110,
#			)
#		)
#	#sgs_dist = vtl.Supra_Glottal_Sequence( dist )
#	#sgs_dist.plot_distributions()
#	#plt.hist( dist[ :, 3 ], label = label )
#	articulatory_distance = [ cosine( avg.reshape(19), x.reshape(19) ) for x in dist ]
#	tsne_points = np.array( [ x for index, x in enumerate( X_embedded ) if labels[ index ] == label ] )
#	tsne_distance = [ mean_squared_error( [np.mean(tsne_points[:,0]), np.mean(tsne_points[:,1] ) ], x ) for x in tsne_points ]
#	print( '' )
#	print( 'Label: {}, median error: {}, mean_error: {}, std: {}'.format( label, np.median(dist_error), np.mean(dist_error), np.std(dist_error) ) )
#	#print( 'Phone: C, median error: {}, mean_error: {}, std: {}'.format( np.median(dist_error_c), np.mean(dist_error_c), np.std(dist_error_c) ) )
#	#print( 'Phone: V, median error: {}, mean_error: {}, std: {}'.format( np.median(dist_error_v), np.mean(dist_error_v), np.std(dist_error_v ) ) )
#	print( 'Mean art. distance: {} +/- {}'.format( np.mean(articulatory_distance), np.std(articulatory_distance) ) )
#	print( 'mean tsne distance: {} +/- {}'.format( np.mean(tsne_distance), np.std(tsne_distance) ) )
#	#print( 'median effort: {} +/- {}'.format( np.median( effort), np.std( effort ) ) )



plt.scatter( X_embedded[ :, 0 ], X_embedded[:, 1] , c = labels )#, c = dist_error )#c = labels )
#for x, label in zip( X_embedded, labels ):
#    plt.text(x[0], x[1], str(label), color="red", fontsize=12)
plt.colorbar()
plt.show()