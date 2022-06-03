


import base64
import VocalTractLab as vtl
import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '-1'
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')
#from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
#from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import one_hot
import matplotlib.pyplot as plt
import requests

from sklearn.metrics import mean_squared_error

from phoneme_recognition import preprocess
from phoneme_classes import sequence_encoder, sequence_decoder
from phoneme_recognition import single_phoneme_recognition_model
#rom agents import Perceptual_Learner_VT_Space
from visual_measurements import get_visual_data

#from tools_mp import multiprocess
from pyMetaheuristic.algorithm import sine_cosine_algorithm
from pyMetaheuristic.algorithm import simulated_annealing
from pyMetaheuristic.algorithm import random_search
from pyMetaheuristic.algorithm import salp_swarm_algorithm
from pyMetaheuristic.algorithm import harris_hawks_optimization
from pyMetaheuristic.algorithm import genetic_algorithm
from pyMetaheuristic.algorithm import differential_evolution
from pyMetaheuristic.algorithm import whale_optimization_algorithm
from pyMetaheuristic.algorithm import flower_pollination_algorithm
from pyMetaheuristic.algorithm import gravitational_search_algorithm
from pyMetaheuristic.algorithm import artificial_bee_colony_optimization
from pyMetaheuristic.algorithm import arithmetic_optimization_algorithm
from pyMetaheuristic.algorithm import cross_entropy_method
from pyMetaheuristic.algorithm import firefly_algorithm
from pyMetaheuristic.algorithm import flow_direction_algorithm
from pyMetaheuristic.algorithm import improved_whale_optimization_algorithm

import json
import time

import argparse
from copy import deepcopy

from tools_io import save, load


#from phoneme_recognition import phoneme_recognition_model

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '-1'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




class EarlyStoppingException( Exception ): pass

#---------------------------------------------------------------------------------------------------------------------------------------------------#
class Optimization_State():
	def __init__(
		self,
		phoneme_identity,
		phoneme_name,
		constriction,
		optimize_glottis,
		optimize_supra_glottal_parameters,
		supra_glottal_base_state,
		glottal_base_state,
		supra_glottal_time_constant,
		glottal_time_constant,
		supra_glottal_duration,
		glottal_offset,
		#acoustic_duration,
		contributes_to_constriction_loss,
		contributes_to_phoneme_loss,
		contributes_to_visual_loss,
		#include_visual_information,
		):
		self.phoneme_identity = phoneme_identity
		self.phoneme_name = phoneme_name
		self.phoneme_state = np.array( [ one_hot( x, 37 ) for x in sequence_encoder( [ phoneme_identity ] ) ] ).reshape( (37,) )
		self.constriction = constriction
		self.optimize_glottis = optimize_glottis
		self.optimize_supra_glottal_parameters = optimize_supra_glottal_parameters
		#print( 'sgbs: ', supra_glottal_base_state )
		#print( isinstance( supra_glottal_base_state, str ) )
		if isinstance( supra_glottal_base_state, str ):
			self.supra_glottal_base_state = vtl.get_shape( supra_glottal_base_state )
		else:
			self.supra_glottal_base_state = vtl.Supra_Glottal_Sequence( np.reshape( np.array( supra_glottal_base_state ), ( 1, 19 ) ) )
		if isinstance( glottal_base_state, str ):
			self.glottal_base_state = vtl.get_shape( glottal_base_state )
		else:
			self.glottal_base_state = vtl.Sub_Glottal_Sequence( np.reshape( np.array( glottal_base_state ), ( 1, 11 ) ) )
		self.supra_glottal_time_constant = supra_glottal_time_constant
		self.glottal_time_constant = glottal_time_constant
		self.supra_glottal_duration = supra_glottal_duration
		self.glottal_offset = glottal_offset
		#self.acoustic_duration = acoustic_duration
		self.contributes_to_constriction_loss = contributes_to_constriction_loss
		self.contributes_to_phoneme_loss = contributes_to_phoneme_loss
		self.contributes_to_visual_loss = contributes_to_visual_loss
		#self.include_visual_information = include_visual_information
		return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	@classmethod
	def from_standard_parameters( 
		cls,
		phoneme,
		supra_glottal_duration,
		include_visual_information,
		state = 'optimize',
		):
		glottal_offset = 0
		if phoneme in [ 'a', 'e', 'i', 'o', 'u', 'E', '2', 'y', 'I', 'O', 'U', '9', 'Y', '6', '@' ]:
			optimize_glottis = False
			constriction = 0
		if phoneme in [ 'f', 's', 'S', 'C', 'x', 'R', 'h' ]:
			optimize_glottis = True
			constriction = 1
		if phoneme in [ 'v', 'z', 'Z', 'j', 'l' ]:
			optimize_glottis = False
			constriction = 1
		if phoneme in [ 'p', 't', 'k' ]:
			optimize_glottis = True
			constriction = 2
		if phoneme in [ 'b', 'd', 'g', 'm', 'n', 'N' ]:
			optimize_glottis = False
			constriction = 2
		if phoneme in [ 'f', 's', 'S', 'C', 'x', ]:
			glottal_offset = -0.01 # -10ms glottal target shift for voiceless fricatives
		if phoneme in [ 'p', 't', ]:
			glottal_offset = 0.05 # +50ms glottal target shift for voiceless plosives p, t
		if phoneme in [ 'k', ]:
			glottal_offset = 0.06 # +60ms glottal target shift for voiceless plosive k
		if state == 'optimize':
			if phoneme in [ 'm', 'n', 'N' ]:
				optimize_supra_glottal_parameters = [
					'HX', 'HY', 'JX', 'JA',
					'LP', 'LD', 'VS', 'VO', 'TCX',
					'TCY', 'TTX', 'TTY', 'TBX',
					'TBY', 'TS1', 'TS2', 'TS3',
				]
			else:
				optimize_supra_glottal_parameters = [
					'HX', 'HY', 'JX', 'JA',
					'LP', 'LD', 'VS', 'TCX',
					'TCY', 'TTX', 'TTY', 'TBX',
					'TBY', 'TS1', 'TS2', 'TS3',
				]
			supra_glottal_base_state = '@'
			contributes_to_constriction_loss = True
		elif state == 'vtl_preset':
			optimize_supra_glottal_parameters = []
			supra_glottal_base_state = phoneme
			contributes_to_constriction_loss = False
		elif state == 'optimized_preset':
			optimize_supra_glottal_parameters = []
			supra_glottal_base_state = select_optimized_preset( path )
			contributes_to_constriction_loss = False
		else:
			raise ValueError( 'state argument is invalid, valid options are: optimize, vtl_preset, optimized_preset' )
		if phoneme in [ 'E', 'I', 'O', 'U', '9', 'Y', 'S', 'Z', 'N' ]:
			phoneme_name = 'cap{}'.format( phoneme )
		else:
			phoneme_name = phoneme
		#supra_glottal_base_state = '@'
		glottal_base_state = 'modal'
		acoustic_duration = 0.15
		supra_glottal_time_constant = 0.012
		glottal_time_constant = 0.012
		contributes_to_phoneme_loss = True,
		return cls(
			phoneme_identity = phoneme,
			phoneme_name = phoneme_name,
			constriction = constriction,
			optimize_glottis = optimize_glottis,
			optimize_supra_glottal_parameters = optimize_supra_glottal_parameters,
			supra_glottal_base_state = supra_glottal_base_state,
			glottal_base_state = glottal_base_state,
			supra_glottal_time_constant = supra_glottal_time_constant,
			glottal_time_constant = glottal_time_constant,
			supra_glottal_duration = supra_glottal_duration,
			glottal_offset = glottal_offset,
			#acoustic_duration = acoustic_duration,
			contributes_to_constriction_loss = contributes_to_constriction_loss,
			contributes_to_phoneme_loss = contributes_to_phoneme_loss,
			contributes_to_visual_loss = include_visual_information,
			)
		#dict(
		#	optimize_glottis = optimize_glottis,
		#	constriction = constriction,
		#	optimize_supra_glottal_parameters = optimize_supra_glottal_parameters,
		#	phoneme_name = phoneme_name,
		#	supra_glottal_duration = supra_glottal_duration,
		#	glottal_duration = glottal_duration,
		#	supra_glottal_base_state = '@',
		#	glottal_base_state = 'modal',
		#	)
#---------------------------------------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------------------------------------#
class Agent():
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def __init__(
		self,
		optimization_states: list,
		phoneme_loss_function,
		phoneme_recognition_model,
		optimization_policy,
		optimization_policy_kw,
		log_dir: str,
		max_synthesis_steps,
		):
		self.optimization_states = optimization_states


		self.phoneme_loss_function = phoneme_loss_function
		self.phoneme_recognition_model = phoneme_recognition_model
		#phoneme_model_save_dir = 'models/RUN_2_tr_KIEL_BITS_te_VTL'
		#self.phoneme_recognition_model = single_phoneme_recognition_model( phoneme_model_save_dir )
		self.optimization_policy = optimization_policy
		self.optimization_policy_kw = deepcopy( optimization_policy_kw )
		self.log_category_data = []
		self.log_loss_data = [
			dict(
				step = -1,
				total_loss = np.inf,
				total_phoneme_loss = np.inf,
				total_visual_loss = np.inf,
				category_losses = { x.phoneme_identity: np.inf for x in self.optimization_states },
				phoneme_losses = { x.phoneme_identity: np.inf for x in self.optimization_states },
				visual_losses = { x.phoneme_identity: np.inf for x in self.optimization_states },
				phoneme_recognition_states = { x.phoneme_identity: None for x in self.optimization_states },
				supra_glottal_sequence = None,
				glottal_sequence = None,
				)
			]
		#self.fmin = [ [ -1, np.inf ] ]
		#self.fmin_states = []
		self.step = 0
		self.max_synthesis_steps = max_synthesis_steps
		self.log_dir = log_dir

		#self.optimization_states = optimization_states
		#print( self.optimization_states )

		if not os.path.exists( self.log_dir ):
			os.makedirs( self.log_dir )

		self.n_phoneme_contributions = len( [ x for x in self.optimization_states if x.contributes_to_phoneme_loss ] )
		self.n_visual_contributions = len( [ x for x in self.optimization_states if x.contributes_to_visual_loss ] )

		self.optimization_parameters = [ y for x in list( self.optimization_states ) for y in x.optimize_supra_glottal_parameters ]

		min_state, max_state = self.get_valid_search_space_range( self.optimization_states )

		#supra_glottal_parameter_info = vtl.get_param_info( 'tract' )
		#supra_glottal_parameter_info.loc[ 'LD', 'min' ] = -0.5
		#supra_glottal_parameter_info.loc[ 'LD', 'max' ] = 2.0

		#self.min_state = supra_glottal_parameter_info.loc[ optimize_supra_glottal_parameters, 'min' ].to_numpy( dtype = float )
		#self.max_state = supra_glottal_parameter_info.loc[ optimize_supra_glottal_parameters, 'max' ].to_numpy( dtype = float )
		#print(self.min_state)
		#print(self.max_state)
		#stop
		#if len( optimize_glottal_parameters ) != 0:
		#	min_glottal_states = vtl.get_param_info( 'glottis' ).loc[ optimize_glottal_parameters, 'min' ].to_numpy( dtype = float )
		#	max_glottal_states = vtl.get_param_info( 'glottis' ).loc[ optimize_glottal_parameters, 'max' ].to_numpy( dtype = float )
		#	self.min_state = np.concatenate( [ self.min_state, min_glottal_states ] )
		#	self.max_state = np.concatenate( [ self.max_state, max_glottal_states ] )
		#	#print( self.min_state.shape )
		#st
		self.supra_glottal_dimension = ( len( self.optimization_states ), vtl.get_constants()[ 'n_tract_params' ] )
		self.glottal_dimension = ( len( self.optimization_states ), vtl.get_constants()[ 'n_glottis_params' ] )

		self.phoneme_segments = []
		for index, x in enumerate( self.optimization_states ):
			self.phoneme_segments.append(
				[
					np.sum( [ x.supra_glottal_duration for x in self.optimization_states[ : index ] ] ) + 0.05 - 0.032,
					np.sum( [ x.supra_glottal_duration for x in self.optimization_states[ : index + 1 ] ] ) + 0.05 + 0.032,
					]
				)

		#print( 'goal state shape: ', self.goal_states.shape )
		self.optimization_policy_kw[ 'min_values' ] = min_state
		self.optimization_policy_kw[ 'max_values' ] = max_state
		return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def __str__( self ):
		log = []
		log.append( 'Artificial vocal learning agent with following parameters:' )
		log.append( ' — Optimization states: {}'.format( self.optimization_states ) )
		log.append( ' — phoneme_loss_function: {}'.format( self.phoneme_loss_function.__class__.__name__ ) )
		log.append( ' — phoneme_recognition_model: {}'.format( self.phoneme_recognition_model.__class__.__name__ ) )
		log.append( ' — optimization_policy: {}'.format( self.optimization_policy.__name__ ) )
		log.append( ' — optimization_policy_kw: {}'.format( self.optimization_policy_kw ) )
		log.append( ' — Category_data: {} entries'.format( len( self.log_category_data ) ) )
		log.append( ' — Loss data: {} entries'.format( len( self.log_loss_data ) ) )
		log.append( ' — Evaluated steps: {}'.format( self.step ) )
		log.append( ' — Stopping after: {} synthesis steps'.format( self.max_synthesis_steps ) )
		log.append( ' — Output log path: {}'.format( self.log_dir ) )
		log.append( ' — supra_glottal_dimension: {}'.format( self.supra_glottal_dimension ) )
		log.append( ' — glottal_dimension: {}'.format( self.glottal_dimension ) )
		log.append( ' — phoneme_segments: {}'.format( self.phoneme_segments ) )
		return '\n'.join( log )
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	@classmethod
	def from_file(
		cls,
		agent_file_path,
		phoneme_loss_function,
		phoneme_recognition_model,
		verbose = True,
		):
		return cls( )
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def to_file( self, agent_file_path, verbose = True ):
		if verbose:
			print( 'Saving agent instance to {}'.format( agent_file_path ) )
			print( 'INFO: Note that the class attributes "phoneme_loss_function" and "phoneme_recognition_model"\n' +
				'were set to None in order to create a serializable object. If you plan to run/continue\n' +
				'the vocal learning simulation after the object is loaded again, these attributes have to be set\n' +
				'explicitly (e.g. use the from_file() functionality or set them directly).'
				)
		self.phoneme_loss_function = None
		self.phoneme_recognition_model = None
		save( self, agent_file_path )
		return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def get_valid_search_space_range(
		self,
		optimization_states,
		):
		supra_glottal_parameter_info = vtl.get_param_info( 'tract' )
		supra_glottal_parameter_info.loc[ 'LD', 'min' ] = -0.5
		supra_glottal_parameter_info.loc[ 'LD', 'max' ] = 2.0

		#optimize_supra_glottal_parameters = [ y for x in list( optimization_states ) for y in x.optimize_supra_glottal_parameters ]

		min_state = supra_glottal_parameter_info.loc[ self.optimization_parameters, 'min' ].to_numpy( dtype = float )
		max_state = supra_glottal_parameter_info.loc[ self.optimization_parameters, 'max' ].to_numpy( dtype = float )
		#print(min_state)
		#print(max_state)
		return min_state, max_state
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def get_motor_score( self, supra_glottal_states, glottal_states ):
		total_articulatory_duration = np.sum( [ x.supra_glottal_duration for x in self.optimization_states ] )
		supra_glottal_motor_score = vtl.Supra_Glottal_Motor_Score.from_supra_glottal_sequence(
			supra_glottal_states,
			synchronous = [ [ 'other' ] ],
			durations = [ [ x.supra_glottal_duration for x in self.optimization_states ] ],
			time_constants = [ [ x.supra_glottal_time_constant for x in self.optimization_states ] ],
		)
		glottal_durations = []
		for index, x in enumerate( self.optimization_states ): # TODO: leads potentially to non equal glottal and subglottal dur. to to shift at end if last offset != 0
			glottal_durations.append(
				np.sum( [ y.supra_glottal_duration for y in self.optimization_states[ : index + 1 ] ] )
				+ x.glottal_offset
				- np.sum( glottal_durations )
				)
		sub_glottal_motor_score = vtl.Sub_Glottal_Motor_Score.from_sub_glottal_sequence(
			glottal_states,
			synchronous = [ [ 'other' ] ],
			durations = [ glottal_durations ],
			time_constants = [ [ x.glottal_time_constant for x in self.optimization_states ] ],
		)
		pressure = vtl.Target_Sequence(
			offsets = [ 8000, 0 ],
			durations = [ total_articulatory_duration - 0.05 , 0.05 ],
			time_constants = [ 0.005, 0.005 ],
			onset_state = 0,
			name = 'PR',
		)
		sub_glottal_motor_score.target_sequences[ 1 ] = pressure
		motor_score = vtl.Motor_Score( supra_glottal_motor_score, sub_glottal_motor_score )
		return motor_score
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def synthesize(
		self,
		supra_glottal_sequence,
		glottal_sequence,
		audio_file_path,
		):
		motor_score_list = self.get_motor_score( supra_glottal_sequence, glottal_sequence )
		vtl.tract_sequence_to_audio(
			motor_score_list,
			[ audio_file_path ],
			save_file = True,
			return_data = False,
			sr = 16000,
			normalize_audio = -1,
			)
		return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def _initialize_run( self ):
		print( 'Running artificial vocal learning simulation:')
		print( 'Optimizing for phoneme identities: {}'.format(
			[ x.phoneme_identity for x in self.optimization_states if len( x.optimize_supra_glottal_parameters ) > 0 ]
			)
		)
		if any( [ x.phoneme_identity for x in self.optimization_states if len( x.optimize_supra_glottal_parameters ) == 0 ] ):
			print( 'In the context: {}'.format( [ x.phoneme_identity for x in self.optimization_states ] ) )
		print( 'Optimizing the following {} parameters:'.format( len( self.optimization_parameters ) ) )
		for index, x in enumerate( self.optimization_states ):
			print( '    State {}:\n    {}'.format( index, x.optimize_supra_glottal_parameters ) )
		print( 'Using following optimization algorithm: {}'.format( self.optimization_policy.__name__ ) )
		print( 'With the following optimization parameters:' )
		for key, val in self.optimization_policy_kw.items():
			print( '	{}: {}'.format( key, val ) )
		print( 'Saving the results to: {}'.format( self.log_dir ) )
		return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def _finalize_run( self, elapsed_time ):
		if len( self.log_loss_data ) > 1:
			supra_glottal_sequence = self.log_loss_data[ -1 ][ 'supra_glottal_sequence' ]
			glottal_sequence = self.log_loss_data[ -1 ][ 'glottal_sequence' ]
			self.synthesize(
				supra_glottal_sequence = supra_glottal_sequence,
				glottal_sequence = glottal_sequence,
				audio_file_path = os.path.join( self.log_dir, 'result.wav' ),
				)
		save( elapsed_time, os.path.join( self.log_dir, 'computation_time.pkl.gzip' ) )
		save( self.log_category_data, os.path.join( self.log_dir, 'log_category_data.pkl.gzip' ) )
		save( self.log_loss_data, os.path.join( self.log_dir, 'log_loss_data.pkl.gzip' ) )
		#save( self.fmin, os.path.join( self.log_dir, 'fmin.pkl.gzip' ) )
		#save( self.fmin_states, os.path.join( self.log_dir, 'fmin_states.pkl.gzip' ) )
		self.to_file( os.path.join( self.log_dir, 'agent.pkl.gzip' ) )
		print( '\nSimulation finished.' )
		return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def run( self ):
		self._initialize_run()
		start = time.time()
		try:
			result = self.optimization_policy( # TODO result = kann weg
				target_function = self.objective_function,
				**self.optimization_policy_kw,
				)
		except EarlyStoppingException:
			print( 'Stopped optimization at step: {}'.format( self.step ) )
		end = time.time()
		elapsed_time = end - start
		self._finalize_run( elapsed_time )
		return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def calculate_phoneme_losses(
		self,
		audio_in,
		#phoneme_goal_state,
		):
		audio_segments = [ audio_in[ int( seg_min * 16000 ) : int( seg_max * 16000 ) ] for seg_min, seg_max in self.phoneme_segments ]
		#main_losses = np.array( [ self.calculate_phoneme_loss( audio_in, x.phoneme_state ) for audio_in, x in zip( audio_segments, self.optimization_states ) ] )
		phoneme_losses = []
		category_losses = []
		phoneme_recognition_states = []
		for audio_segment, x in zip( audio_segments, self.optimization_states ):
			X_input = preprocess( audio_in = audio_segment )
			y_pred = self.phoneme_recognition_model.predict( X_input )[0]
			phoneme_recognition_states.append( y_pred )
			y_true = x.phoneme_state
			phoneme_losses.append( self.phoneme_loss_function( y_true, y_pred ).numpy() )
			y_pred_category = np.argmax( y_pred )
			y_true_category = np.argmax( y_true )
			#print( 'y_true: ', y_true )
			#print( 'y_pred_category: ', y_pred_category )
			#stop
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
	def calculate_constriction_loss_old(
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
	def calculate_constriction_loss(
		self,
		supra_glottal_states,
		):
		constriction_loss = []
		for state, x in zip( supra_glottal_states, self.optimization_states ):
			if x.contributes_to_constriction_loss:
				tube_state = vtl.tract_sequence_to_tube_states( state )[0]
				if tube_state.constriction != x.constriction:
					constriction_loss.append( 100 )
				else:
					if x.constriction in [ 1, 2 ]:
						if tube_state.has_precise_constriction():
							constriction_loss.append( 0 )
						else:
							constriction_loss.append( 50 )
					else:
						constriction_loss.append( 0 )
			else:
				constriction_loss.append( 0 )
		return np.sum( constriction_loss )
#---------------------------------------------------------------------------------------------------------------------------------------------------#
	def objective_function( self, articulatory_states ):
		if self.step >= self.max_synthesis_steps:
			raise EarlyStoppingException


		supra_glottal_states = [ x.supra_glottal_base_state for x in self.optimization_states ]
		#print( 'sgs: \n', supra_glottal_states )
		#print( len( articulatory_states ) )
		prev_index = 0
		for state_index, x in enumerate( self.optimization_states ):
			#print( 'prev_index: ', prev_index )
			parameters = x.optimize_supra_glottal_parameters
			parameters_queries = articulatory_states[ prev_index : prev_index + len( parameters ) ]
			for param_index, parameter in enumerate( parameters ):
				supra_glottal_states[ state_index ].states.loc[ 0, parameter ] = parameters_queries[ param_index ]
			prev_index = prev_index + len( parameters )
		#print( np.array( [ x.states.to_numpy() for x in supra_glottal_states ] ) )


		constriction_loss = self.calculate_constriction_loss( supra_glottal_states )
		if constriction_loss != 0:
			return constriction_loss
		visual_losses = self.calculate_visual_losses( supra_glottal_states )



		supra_glottal_sequence = vtl.Supra_Glottal_Sequence(
			np.array( [ x.states.to_numpy() for x in supra_glottal_states ] ).reshape( self.supra_glottal_dimension )
			)
		#print(supra_glottal_sequence)


		#tube_states = vtl.tract_sequence_to_tube_states( supra_glottal_sequence )
		#if tube_states[0].constriction != self.constraints[0][ 'constriction' ]:
		#	return 100

		#if self.include_visual_information:
		#	opt_visual_data = [
		#		supra_glottal_states.states.loc[ 0, 'JA' ],
		#		supra_glottal_states.states.loc[ 0, 'LP' ],
		#		supra_glottal_states.states.loc[ 0, 'LD' ],
		#	]
		#	measured_visual_data = get_visual_data( self.phoneme )
		#	visual_loss = mean_squared_error( opt_visual_data, measured_visual_data )
		#	if visual_loss >= 10:
		#		return visual_loss
		#else:
		#	visual_loss = 0
		#effort_loss = 0


		#if isinstance( self.succeeding_state, vtl.Supra_Glottal_Sequence ):
		#	#print(supra_glottal_states.states)
		#	#print(self.succeeding_state.states)
		#	#print( supra_glottal_states.states.iloc[ 0, : ].to_numpy() )
		#	#print( self.succeeding_state.states.iloc[ 0, : ].to_numpy() )
		#	effort_loss = mean_squared_error( supra_glottal_states.states.iloc[ 0, : ].to_numpy(), self.succeeding_state.states.iloc[ 0, : ].to_numpy() )
		#	if effort_loss >= 1:
		#		return effort_loss + 10
		#else:
		#	effort_loss = 0
		#print( supra_glottal_states )
		#stop

		

		#glottal_parameters = articulatory_states[ len( self.optimize_supra_glottal_parameters ) : ]
		glottal_states = [ x.glottal_base_state for x in self.optimization_states ]
		#if len( self.optimize_glottal_parameters ) != 0:
		#	glottal_states.states[ 'XB' ] = glottal_parameters[ self.optimize_glottal_parameters.index( 'XB' ) ]
		#	glottal_states.states[ 'XT' ] = glottal_parameters[ self.optimize_glottal_parameters.index( 'XB' ) ]
		#	glottal_states.states[ 'CA' ] = glottal_parameters[ self.optimize_glottal_parameters.index( 'XB' ) ]
		#	if glottal_parameters[ self.optimize_glottal_parameters.index( 'RA' ) ] <= 0:
		#		glottal_states.states[ 'RA' ] = 0.0
		#	else:
		#		glottal_states.states[ 'RA' ] = 1.0
		glottal_sequence = vtl.Sub_Glottal_Sequence(
			np.array( [ x.states.to_numpy() for x in glottal_states ] ).reshape( self.glottal_dimension )
			)
		#print(glottal_sequence)

		
		motor_score_list = self.get_motor_score( supra_glottal_sequence, glottal_sequence )
		audio_signal = vtl.tract_sequence_to_audio( motor_score_list, save_file = False, return_data = True, sr = 16000 )[0]
		#plt.plot( query[0] )
		#plt.show()
		#stop
		#audio_in = query[0][ int( 0.0 * 16000 ) : int( 0.15 * 16000 ) ] # 0.0, 0.15 for vowels, cons: 0.075, 0.225

		phoneme_losses, category_losses, phoneme_recognition_states = self.calculate_phoneme_losses( audio_signal )

		#audio_segments = [ query[0][ int( seg_min * 16000 ) : int( seg_max * 16000 ) ] for seg_min, seg_max in self.phoneme_segments ]
		#main_losses = np.array( [ self.calculate_phoneme_loss( audio_in, x.phoneme_state ) for audio_in, x in zip( audio_segments, self.optimization_states ) ] )
		#phoneme_losses = main_losses[ :, 0 ]
		#category_losses = main_losses[ :, 1 ]

		total_phoneme_loss = np.sum( phoneme_losses )
		total_visual_loss = np.sum( visual_losses )

		total_loss = total_phoneme_loss + total_visual_loss

		if np.sum( category_losses ) == 0:
			self.log_category_data.append(
				dict(
					step = self.step,
					total_loss = total_loss,
					total_phoneme_loss = total_phoneme_loss,
					total_visual_loss = total_visual_loss,
					category_losses = { x.phoneme_identity: y for x, y in zip( self.optimization_states, category_losses ) },
					phoneme_losses = { x.phoneme_identity: y for x, y in zip( self.optimization_states, phoneme_losses ) },
					visual_losses = { x.phoneme_identity: y for x, y in zip( self.optimization_states, visual_losses ) },
					phoneme_recognition_states = { x.phoneme_identity: y for x, y in zip( self.optimization_states, phoneme_recognition_states ) },
					supra_glottal_sequence = supra_glottal_sequence,
					glottal_sequence = glottal_sequence,
					)
				)

		if total_loss < self.log_loss_data[ -1 ][ 'total_loss' ]:
			self.log_loss_data.append(
				dict(
					step = self.step,
					total_loss = total_loss,
					total_phoneme_loss = total_phoneme_loss,
					total_visual_loss = total_visual_loss,
					category_losses = { x.phoneme_identity: y for x, y in zip( self.optimization_states, category_losses ) },
					phoneme_losses = { x.phoneme_identity: y for x, y in zip( self.optimization_states, phoneme_losses ) },
					visual_losses = { x.phoneme_identity: y for x, y in zip( self.optimization_states, visual_losses ) },
					phoneme_recognition_states = { x.phoneme_identity: y for x, y in zip( self.optimization_states, phoneme_recognition_states ) },
					supra_glottal_sequence = supra_glottal_sequence,
					glottal_sequence = glottal_sequence,
					)
				)
			#self.fmin.append( [ self.step, ftmp, total_phoneme_loss, visual_loss, effort_loss, phoneme_losses ] )
			#self.fmin_states.append( [ supra_glottal_sequence, glottal_sequence ] )
			log = []
			log.append( 'Step: {: <{}}'.format( self.step, len( str( self.max_synthesis_steps ) ) + 1 ) )
			#log.append( 'phoneme loss: {: <{}}'.format( f'{total_phoneme_loss:.5f}', 10 ) )
			if self.n_phoneme_contributions > 1:
				log.append(
					'phoneme loss: {: <{}}'.format( f'{total_phoneme_loss:.5f}', 10 ) +
					'( ' + 
					', '.join( [ '{}: {: <{}}'.format( x.phoneme_identity, f'{y:.3f}', 7 ) for x, y in zip( self.optimization_states, phoneme_losses ) ] ) +
					')'
					)
				log.append( 
					'category loss: '.format( category_losses ) +
					'( ' + 
					', '.join( [ '{}: {}'.format( x.phoneme_identity, y ) for x, y in zip( self.optimization_states, category_losses ) ] ) +
					' )'
					)
			else:
				log.append( 'phoneme loss: {: <{}}'.format( f'{total_phoneme_loss:.5f}', 10 ) )
				log.append( 'category loss: {}'.format( category_losses[0] ) )
			if self.n_visual_contributions == 1:
				log.append( 'visual loss: {: <{}}'.format( f'{total_visual_loss:.5f}', 10 ) )
			elif self.n_visual_contributions > 1:
				log.append(
					'visual loss: {: <{}}'.format( f'{total_visual_loss:.5f}', 9 ) +
					'(' + 
					', '.join( [ '{}: {: <{}}'.format( x.phoneme_identity, f'{y:.5f}', 10 ) for x, y in zip( self.optimization_states, visual_losses ) ] ) +
					')'
					)
			#log.append( 'effort loss: {: <{}}'.format( f'{effort_loss:.5f}', 10 ) )
			#log.append( 'category loss: {}'.format(category_losses) )
			#print( 'Synthesis step: {} — phoneme loss: {} () — visual_loss: {} — effort loss: {}'.format(
			#	self.step, total_phoneme_loss, visual_loss, effort_loss
			#	)
			#)
			print( ' — '.join( log ) )
			#print( 'new min' )
			#print( supra_glottal_states )
			#print( glottal_states )
		self.step += 1
		return total_loss
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def demo(
	optimize_units,
	synthesis_steps = 1000,
	include_visual_information = False,
	runs = [ x for x in range( 0, 5 ) ],
	out_path = 'demo_results/',
	):
	optimization = dict(
		optimization_policy = whale_optimization_algorithm,
		optimization_policy_kw = dict(
			hunting_party = 200, # 200 for vowels #300 for cons currently
			iterations = 500000,
			spiral_param = 0.5,
			verbose = False,
			),
		name = 'woa',
		)
	cce = CategoricalCrossentropy()
	phoneme_recognition_model = single_phoneme_recognition_model()
	simulations = []
	for unit in optimize_units:
		optimization_states = []
		for index, phoneme in enumerate( unit ):
			if index == 0:
				if len( unit ) == 1:
					supra_glottal_duration = 0.2
				else:
					supra_glottal_duration = 0.05
				state = 'optimize'
			else:
				supra_glottal_duration = 0.15
				state = 'vtl_preset'
			optimization_states.append(
				Optimization_State.from_standard_parameters(
					phoneme = phoneme,
					supra_glottal_duration = supra_glottal_duration,
					include_visual_information = include_visual_information,
					state = state,
					)
				)
		for run in runs:
			simulations.append(
				Agent(
					optimization_states = optimization_states,
					phoneme_loss_function = cce,
					phoneme_recognition_model = phoneme_recognition_model,
					optimization_policy = optimization[ 'optimization_policy' ],
					optimization_policy_kw = optimization[ 'optimization_policy_kw' ],
					log_dir = os.path.join( 
						out_path, 
						'{}/{}/{}'.format(
							optimization[ 'name' ],
							'_'.join( [ x.phoneme_name for x in optimization_states ] ),
							run,
							),
						),
					max_synthesis_steps = synthesis_steps,
					)
				)
	for simulation in simulations:
		simulation.run()
	return
#---------------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
	parser = argparse.ArgumentParser( description='Process some integers.' )
	parser.add_argument( '--range', metavar='range', type=int, nargs='+', help='range integers' )
	args = parser.parse_args()

	#demo( optimize_units = [ [ 'b', 'a' ], ], runs = [ x for x in range( 0, 10 ) ], synthesis_steps = 1000, out_path = 'demo_test/CONSONANT_VTL_PRESET_VOWELS/' )
	demo(
		optimize_units = [ [ 'a' ], [ 'e' ], [ 'i' ], [ 'o' ], [ 'u' ], [ 'E' ], [ '2' ], [ 'y' ], ],
		runs = [ x for x in range( args.range[ 0 ], args.range[ 1 ] ) ],
		synthesis_steps = 1000,
		out_path = 'results/VOWELS_VISUAL/',
		include_visual_information = True,
		)
	#demo(
	#	optimize_units = [ [ 'd', 'i' ], ],
	#	runs = [ x for x in range( args.range[ 0 ], args.range[ 1 ] ) ],
	#	synthesis_steps = 1000,
	#	out_path = 'demo_run/CONSONANT_EXTCONSTR_TEST/',
	#	include_visual_information = False,
	#	)
	#demo_print( optimize_units = [ [ 'z', 'i' ], [ 'n', 'i' ] ], runs = [ 0 ], synthesis_steps = 1000, out_path = 'demo_test/CONSONANT_VTL_PRESET_VOWELS/' )
	#ag = load( 'demo/CONSONANT_VTL_PRESET_VOWELS/woa/e/15/agent.pkl.gzip')
	#print(ag)
	#category_data = load( 'demo/CONSONANT_VTL_PRESET_VOWELS/woa/e/15/log_category_data.pkl.gzip')
	#print( category_data[ 0 ][ 'supra_glottal_sequence' ] )
	#stop
	#array = np.array( [ x[ 'supra_glottal_sequence' ].states.to_numpy() for x in category_data ] ).reshape( (len(category_data), 19 ) )
	#sgs = vtl.Supra_Glottal_Sequence( array )
	#sgs.plot_distributions()
	#avg = vtl.Supra_Glottal_Sequence( np.reshape( np.median( array, axis = 0 ), (1,19) ) )
	#glt = category_data[0][ 'glottal_sequence' ]
	#ag.synthesize( avg, glt, 'testboy.wav' )