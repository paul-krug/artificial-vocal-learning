




def tri_phoneme_loss(
	agent,
	audio_list, #list of dicts
	):
	phoneme_losses
	X_input = preprocess_list( audio_in = audio_list )
	y_pred = agent.phoneme_recognition_model.predict( X_input, verbose = 0 )[0]
	y_true = np.array( [ x.phoneme_state for x in self.optimization_states ] )
	phoneme_losses.append( self.phoneme_loss_function( y_true, y_pred ).numpy() )
	category_losses = [ 0 if np.argmax( y_p ) == np.argmax( y_t ) else 1 for y_p, y_t in zip( y_pred, y_true ) ]
	return phoneme_losses, category_losses, y_pred
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def calculate_phoneme_losses(
		self,
		audio_in,
		):
		audio_segments = [ audio_in[ int( seg_min * 16000 ) : int( seg_max * 16000 ) ] for seg_min, seg_max in self.phoneme_segments ]
		phoneme_losses = []
		category_losses = []
		phoneme_recognition_states = []
		for audio_segment, x in zip( audio_segments, self.optimization_states ):
			X_input = preprocess( audio_in = audio_segment )
			y_pred = self.phoneme_recognition_model.predict( X_input, verbose=0 )[0]
			phoneme_recognition_states.append( y_pred )
			y_true = x.phoneme_state
			phoneme_losses.append( self.phoneme_loss_function( y_true, y_pred ).numpy() )
			y_pred_category = np.argmax( y_pred )
			y_true_category = np.argmax( y_true )
			#print( 'y_true: ', y_true )
			#print( 'y_pred_category: ', y_pred_category )
			if y_pred_category == y_true_category:
				category_losses.append( 0 )
			else:
				category_losses.append( 1 )
		return phoneme_losses, category_losses, phoneme_recognition_states
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def calculate_visual_losses(
		self,
		supra_glottal_states,
		):
		visual_losses = []
		for state, x in zip( supra_glottal_states, self.optimization_states ):
			if x.contributes_to_visual_loss:
				opt_visual_data = [
					state.states.loc[ 0, 'JA' ],
					state.states.loc[ 0, 'LP' ],
					state.states.loc[ 0, 'LD' ],
				]
				measured_visual_data = get_visual_data( x.phoneme_identity )
				visual_losses.append( mean_squared_error( opt_visual_data, measured_visual_data ) )
			else:
				visual_losses.append( 0 )
		return visual_losses
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def calculate_constriction_loss(
		self,
		supra_glottal_states,
		):
		constriction_loss = np.sum( [
				100
				if x.contributes_to_constriction_loss and ( vtl.tract_sequence_to_tube_states( state )[0].constriction != x.constriction )
				else 0
				for state, x in zip( supra_glottal_states, self.optimization_states )
				]
			)
		return constriction_loss
#---------------------------------------------------------------------------------------------------------------------------------------------------#