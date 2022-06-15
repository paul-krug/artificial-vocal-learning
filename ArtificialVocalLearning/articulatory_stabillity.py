import VocalTractLab as vtl
import numpy as np
import pandas as pd
import os
from tools_io import save, load
import matplotlib.pyplot as plt







if __name__ == '__main__':
	import vocal_learning
	from vocal_learning import Agent, Optimization_State
	agent = load( 'results/VOWELS_VISUAL/woa/u/0/agent.pkl.gzip' )
	agent.phoneme_loss_function = vocal_learning.CategoricalCrossentropy()
	agent.phoneme_recognition_model = vocal_learning.single_phoneme_recognition_model()
	#sgs = vtl.get_shapes( [ 'a' ] )
	glt = vtl.get_shapes( [ 'modal' ] )
	limit = np.abs( agent.optimization_policy_kw[ 'max_values' ] - agent.optimization_policy_kw[ 'min_values' ] )
	n_repititions = 100
	data = []
	for noise in [ 0.01 * n for n in range( 1, 11 ) ]:
		sgs_list = []
		for repitition in range( 0, n_repititions ):
			sgs = vtl.get_shapes( [ 'a' ] )
			for index, param in enumerate( agent.optimization_parameters ):
				sgs.states.loc[ 0, param ] += np.random.normal( 0, noise * limit[ index ] )
			sgs_list.append( sgs )
		motor_score_list = [ agent.get_motor_score( entry, glt ) for entry in sgs_list ]
		#print( motor_score_list )
		audio_file_list = vtl.tract_sequence_to_audio(
			motor_score_list,
			#[ audio_file_path ],
			save_file = False,
			return_data = True,
			sr = 16000,
			normalize_audio = -1,
			)
		phoneme_losses = [ agent.calculate_phoneme_losses( audio_file_list[x] )[2][0][0] for x in range( 0, n_repititions ) ]
		data.append( phoneme_losses )
	plt.boxplot( data )
	plt.show()

