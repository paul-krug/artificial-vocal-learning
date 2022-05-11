



import VocalTractLab as vtl
import numpy as np
from itertools import chain




vowels = [ 'a', 'e', 'i', 'o', 'u', 'E', 'I', 'O', 'U', '2', '9', 'y', 'Y', '@', '6' ]
consonants = vtl.single_consonants()
consonants.remove( 'T' )
consonants.remove( 'D' )
consonants.remove( 'r' )
classes = np.array( [ x for x in chain( vowels, consonants ) ] )




#---------------------------------------------------------------------------------------------------------------------------------------------------#
def sequence_encoder( phoneme_sequence ):
	labels = []
	for phoneme in phoneme_sequence:
		labels.append( np.where( classes == phoneme )[0][0] )
	return np.array( labels )
#---------------------------------------------------------------------------------------------------------------------------------------------------#
def sequence_decoder( label_sequence ):
	phonemes = []
	for label in label_sequence:
		phonemes.append( classes[ label ] )
	return np.array( phonemes )
#---------------------------------------------------------------------------------------------------------------------------------------------------#