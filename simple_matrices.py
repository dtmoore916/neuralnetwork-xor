#!/usr/local/bin/python

import numpy
from numpy.linalg import inv

# input | output
# --------------
# 0, 0  | 0
# 0, 1  | 1
# 1, 0  | 1
# 1, 1  | 0

#      W1
#      W2  H1   Wx1
#
#  I   W3
#      W4  H2   Wx2   O
#  I
#      W5  H3   Wx3
#      W6
#

# sigmoid function
def sigmoid(x):
	return 1/(1+numpy.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))


def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with numpy.errstate(divide='ignore', invalid='ignore'):
        c = numpy.true_divide( a, b )
        c[ ~ numpy.isfinite( c )] = 0  # -inf inf NaN
    return c


input_matrix = numpy.zeros((4, 2))
input_matrix[0][0] = 0; input_matrix[0][1] = 0
input_matrix[1][0] = 0; input_matrix[1][1] = 1
input_matrix[2][0] = 1; input_matrix[2][1] = 0
input_matrix[3][0] = 1; input_matrix[3][1] = 1
#input_matrix[0][0] = 0; input_matrix[0][1] = 0
#input_matrix[1][0] = 0; input_matrix[1][1] = 0
#input_matrix[2][0] = 0; input_matrix[2][1] = 0
#input_matrix[3][0] = 1; input_matrix[3][1] = 1

output_matrix = numpy.zeros((4, 1))
output_matrix[0][0] = 0
output_matrix[1][0] = 1
output_matrix[2][0] = 1
output_matrix[3][0] = 0


weights_hidden_matrix = numpy.zeros((2, 3))
weights_hidden_matrix[0][0] = .8; weights_hidden_matrix[0][1] = .4; weights_hidden_matrix[0][2] = .3
weights_hidden_matrix[1][0] = .2; weights_hidden_matrix[1][1] = .9; weights_hidden_matrix[1][2] = .5

weights_output_matrix = numpy.zeros((3, 1))
weights_output_matrix[0][0] = .3;
weights_output_matrix[1][0] = .5;
weights_output_matrix[2][0] = .9;

for i in range(0, 3):
	#print '############################################'
	#print '### Forward Propagation for input (1, 1) ###'
	#print '############################################'
	Hsum = numpy.dot(input_matrix, weights_hidden_matrix)
	H = sigmoid(Hsum)

	OsumM = numpy.dot(H, weights_output_matrix)
	outputM = sigmoid(OsumM)

	#Back Propation starts
	#print '############################################'
	#print '### Back Propagation ###'
	#print '############################################'
	#print
	error_matrix = output_matrix - outputM
	#print
	#print('{} error={}'.format(i, numpy.sum(error_matrix, axis=0)))
	#print
	sigmoid_prime_osum = sigmoid_prime(OsumM)
	DeltaOutputSumMatrix = sigmoid_prime_osum * error_matrix

	DeltaWeightsMatrix = DeltaOutputSumMatrix / H
	DeltaWeightsMatrix = numpy.sum(DeltaWeightsMatrix, axis=0)
	DeltaWeightsMatrix = numpy.reshape(DeltaWeightsMatrix, (3,1))
	UpdatedWeightsMatrix = weights_output_matrix + DeltaWeightsMatrix
	#print('{}'.format(DeltaWeightsMatrix))
	#print('{}'.format(UpdatedWeightsMatrix))
	#print
	#print

	#weights_output_matrix_reshaped_invert_for_division = 1 / weights_output_matrix.transpose()
	#Hdelta_hidden_sum = DeltaOutputSumMatrix * weights_output_matrix_reshaped_invert_for_division * sigmoid_prime(Hsum)

	#Delta_weights_input = numpy.dot(div0(1, input_matrix.transpose()), Hdelta_hidden_sum)

	#New_weights_input = weights_hidden_matrix.copy()
	#New_weights_input = New_weights_input + Delta_weights_input

	#weights_hidden_matrix = New_weights_input;
	weights_output_matrix = UpdatedWeightsMatrix;

	print('output={}'.format(outputM))
	print

