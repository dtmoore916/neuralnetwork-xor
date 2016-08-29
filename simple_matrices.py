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

output_matrix = numpy.zeros((4, 1))
output_matrix[0][0] = 0
output_matrix[1][0] = 1
output_matrix[2][0] = 1
output_matrix[3][0] = 0


weights_hidden_matrix = numpy.zeros((2, 3))
# [Weight into H1, Weight into H2, Weight into H3]
# [Weight into H1, Weight into H2, Weight into H3]
weights_hidden_matrix[0][0] = .8; weights_hidden_matrix[0][1] = .4; weights_hidden_matrix[0][2] = .3
weights_hidden_matrix[1][0] = .2; weights_hidden_matrix[1][1] = .9; weights_hidden_matrix[1][2] = .5

#In1    [I1 I2]   H1 H2 H3           H1            H2           H3
#In2    [I3 I4]  [w1 w2 w3]      [I1*w1+I2*w4  I1*w2+I2*w5  I1*w3+I2*w6]  => row is H1, H2, H3 input sum for the hidden row. (In1)
#In3    [I5 I6]  [w4 w5 w6]  =   [I3*w1+I4*w4  I3*w2+I4*w5  I3*w3+I4*w6]  => row is H1, H2, H3 input sum for the hidden row. (In2)
#In4    [I7 I8]                  [I5*w1+I6*w4  I5*w2+I6*w5  I5*w3+I6*w6]  => row is H1, H2, H3 input sum for the hidden row. (In3)
#                                [I7*w1+I8*w4  I7*w2+I8*w5  I7*w3+I8*w6]  => row is H1, H2, H3 input sum for the hidden row. (In4)

weights_output_matrix = numpy.zeros((3, 1))
weights_output_matrix[0][0] = .3;
weights_output_matrix[1][0] = .5;
weights_output_matrix[2][0] = .9;


print 'Input:'
print input_matrix
print
print 'Hidden Weights:'
print weights_hidden_matrix
print
print 'Output Weights:'
print weights_output_matrix
print

# Calculate the forward and backpropation result for just input: 1, 1 for comprehension
#Forward Propagation starts
print '############################################'
print '### Forward Propagation for input (1, 1) ###'
print '############################################'
print

Hsum = numpy.dot(input_matrix, weights_hidden_matrix)
H = sigmoid(Hsum)
print 'Hidden Values'
print H
print
print

OsumM = numpy.dot(H, weights_output_matrix)
outputM = sigmoid(OsumM)
print 'Output Value'
print outputM
print
print

#Back Propation starts
print '############################################'
print '### Back Propagation ###'
print '############################################'
print
error_matrix = output_matrix - outputM
sigmoid_prime_osum = sigmoid_prime(OsumM)
DeltaOutputSumMatrix = sigmoid_prime_osum * error_matrix

print 'Error Matrix:'
print error_matrix
print
print 'Sigmoid Prime of OutputSum:'
print sigmoid_prime_osum
print
print 'DeltaOutputSum: sigmoid_prime_osum * error_matrix'
print DeltaOutputSumMatrix
print

DeltaWeightsMatrix = DeltaOutputSumMatrix / H
DeltaWeightsMatrix = numpy.sum(DeltaWeightsMatrix, axis=0) # Combine all my deltas for each input
DeltaWeightsMatrix = numpy.reshape(DeltaWeightsMatrix, (3,1))
print 'DeltaOutputSumMatrix:'
print DeltaOutputSumMatrix
print
print 'Hidden Values:'
print H
print
print 'DeltaOutputSumMatrix / Hidden Values:'
print '*Matrix of delta weights.'
print '*Rows are the inputs'
print '*Columns are the delta weights per synapses'
print '*3 synapses going to the single output'
print DeltaOutputSumMatrix / H
print
print 'Combine all the delta weights per synapse'
print 'Reshaped to (3,1) to combine to the output weights matrix'
print DeltaWeightsMatrix
print

UpdatedWeightsMatrix = weights_output_matrix + DeltaWeightsMatrix

print 'NewOutputWeights:'
print UpdatedWeightsMatrix
print
print

#                  ( How much each hidden layer affects DeltaOutputSum) * rate of change of input of H
weights_output_matrix_reshaped_invert_for_division = 1 / weights_output_matrix.transpose()
Hdelta_hidden_sum = DeltaOutputSumMatrix * weights_output_matrix_reshaped_invert_for_division * sigmoid_prime(Hsum)

print 'DeltaHiddenSum for neuron per input:'
print Hdelta_hidden_sum
print

#     Should be Hdelta_hidden_sum / Input = delta weights
#     1) transpose my input so the matrix multiplication will
#        work out to do this calculation for every input and add them
#        together.
#     2) take 1 / input and multiple since we need to be dividing by input
Delta_weights_input = numpy.dot(div0(1, input_matrix.transpose()), Hdelta_hidden_sum)
print 'Delta input weights:'
print Delta_weights_input
print

#
#
#     Should be Hdelta_hidden_sum / Input = delta weigths
#       But since there are 0 inputs I get NANs and divide by 1 doesn't matter
#       I guess divide  by 0 should = 0 update to the weight since that input
#       is not impacting anything.
#
#       Code to handle divide by 0
#       http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
#
#                                       Hdelta_hidden_sum (per each input)
#                                  H1          H2          H3
#    I1  I2  I3  I4          [[-0.12246662 -0.07347997 -0.04082221]
# [[ 0.  0.  1.  1.]          [ 0.03954961  0.01970161  0.01251674]
#  [ 0.  1.  0.  1.]]         [ 0.03666196  0.02470696  0.01396589]
#                             [-0.0886695  -0.04554026 -0.03215686]]
#
#[[-0.05200754 -0.02083331 -0.01819096]
# [-0.04911989 -0.02583865 -0.01964011]]

New_weights_input = weights_hidden_matrix.copy()
New_weights_input = New_weights_input + Delta_weights_input
print 'New input weights:'
print New_weights_input
print


#Forward Propagation starts
print '############################################'
print '### Forward Propagation for input (1, 1) ###'
print '############################################'
print

Hsum = numpy.dot(input_matrix, New_weights_input)
H = sigmoid(Hsum)
print 'Input'
print input_matrix
print
print 'Weights'
print New_weights_input
print
print 'Hidden Values'
print H
print
print

OsumM = numpy.dot(H, UpdatedWeightsMatrix)
outputM = sigmoid(OsumM)
print 'Weights Output'
print UpdatedWeightsMatrix
print
print 'Output Value'
print outputM
print



#Output Value
#[[ 0.59223702]
# [ 0.61751077]
# [ 0.60540027]
# [ 0.62864899]]
