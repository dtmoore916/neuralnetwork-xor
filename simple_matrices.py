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

I7 = input_matrix[3][0]
I8 = input_matrix[3][1]
H1sum = I7 * weights_hidden_matrix[0][0] + I8 * weights_hidden_matrix[1][0]
H2sum = I7 * weights_hidden_matrix[0][1] + I8 * weights_hidden_matrix[1][1]
H3sum = I7 * weights_hidden_matrix[0][2] + I8 * weights_hidden_matrix[1][2]
H1 = sigmoid(H1sum)
H2 = sigmoid(H2sum)
H3 = sigmoid(H3sum)
H = sigmoid(Hsum)
print 'Input'
print input_matrix
print
print 'Weights'
print weights_hidden_matrix
print
print 'Hidden Values'
print H
print
print

Osum = H1 * weights_output_matrix[0][0] + H2 * weights_output_matrix[1][0] + H3 * weights_output_matrix[2][0]
output = sigmoid(Osum)

OsumM = numpy.dot(H, weights_output_matrix)
outputM = sigmoid(OsumM)
print 'Hidden Values'
print H
print
print 'Weights Output'
print weights_output_matrix
print
print 'Output Value'
print outputM
print

#Back Propation starts
print '############################################'
print '### Back Propagation ###'
print '############################################'
print
target_value = 0
error_amount = target_value - output
rate_of_change_at_sum = sigmoid_prime(Osum)
DeltaOutputSum = rate_of_change_at_sum * error_amount # moved our output sum by the error amount

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

delta_weights_output = weights_output_matrix.copy()
updated_weights_output = weights_output_matrix.copy()
delta_weights_output[0][0] = DeltaOutputSum / H1;
delta_weights_output[1][0] = DeltaOutputSum / H2;
delta_weights_output[2][0] = DeltaOutputSum / H3;
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

updated_weights_output[0][0] = weights_output_matrix[0][0] + delta_weights_output[0][0]
updated_weights_output[1][0] = weights_output_matrix[1][0] + delta_weights_output[1][0]
updated_weights_output[2][0] = weights_output_matrix[2][0] + delta_weights_output[1][0]
UpdatedWeightsMatrix = weights_output_matrix + DeltaWeightsMatrix

print 'NewOutputWeights:'
print UpdatedWeightsMatrix
print
print
print


#                  ( How much each hidden layer affects DeltaOutputSum) * rate of change of input of H
H1delta_hidden_sum = DeltaOutputSum / weights_output_matrix[0][0] * sigmoid_prime(H1sum)
H2delta_hidden_sum = DeltaOutputSum / weights_output_matrix[1][0] * sigmoid_prime(H2sum)
H3delta_hidden_sum = DeltaOutputSum / weights_output_matrix[2][0] * sigmoid_prime(H3sum)

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






#delta_weights_input = weights_hidden_matrix.copy()
#new_weights_input = weights_hidden_matrix.copy()
## [Weight into H1, Weight into H2, Weight into H3]
## [Weight into H1, Weight into H2, Weight into H3]
#delta_weights_input[0][0] = H1delta_hidden_sum / I7; delta_weights_input[0][1] = H2delta_hidden_sum / I7; delta_weights_input[0][2] = H3delta_hidden_sum / I7
#delta_weights_input[1][0] = H1delta_hidden_sum / I8; delta_weights_input[1][1] = H2delta_hidden_sum / I8; delta_weights_input[1][2] = H3delta_hidden_sum / I8
#
#new_weights_input[0][0] += delta_weights_input[0][0]; new_weights_input[0][1] += delta_weights_input[0][1]; new_weights_input[0][2] += delta_weights_input[0][2]
#new_weights_input[1][0] += delta_weights_input[1][0]; new_weights_input[1][1] += delta_weights_input[1][1]; new_weights_input[1][2] += delta_weights_input[1][2]
#
#
#print 'DeltaInputWeights:'
#print delta_weights_input
#print
#print 'OldInputWeights:'
#print weights_hidden_matrix
#print
#print 'NewInputWeights:'
#print new_weights_input
#print
#


#Forward Propagation starts
print '############################################'
print '### Forward Propagation for input (1, 1) ###'
print '############################################'
print
#I7 = input_matrix[3][0]
#I8 = input_matrix[3][1]
#H1sum = I7 * new_weights_input[0][0] + I8 * new_weights_input[1][0]
#H2sum = I7 * new_weights_input[0][1] + I8 * new_weights_input[1][1]
#H3sum = I7 * new_weights_input[0][2] + I8 * new_weights_input[1][2]
#H1 = sigmoid(H1sum)
#H2 = sigmoid(H2sum)
#H3 = sigmoid(H3sum)
#print 'Hidden Values'
#print H1
#print H2
#print H3
#print
#Osum = H1 * updated_weights_output[0][0] + H2 * updated_weights_output[1][0] + H3 * updated_weights_output[2][0]
#output = sigmoid(Osum)
#print 'Output Value'
#print output
#print



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

