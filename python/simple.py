#!/usr/local/bin/python

import numpy

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


input_matrix = numpy.zeros((4, 2))
input_matrix[0][0] = 0; input_matrix[0][1] = 0
input_matrix[1][0] = 0; input_matrix[1][1] = 1
input_matrix[2][0] = 1; input_matrix[2][1] = 0
input_matrix[3][0] = 1; input_matrix[3][1] = 1

weights_hidden_matrix = numpy.zeros((2, 3))
# [Weight into H1, Weight into H2, Weight into H3]
# [Weight into H1, Weight into H2, Weight into H3]
weights_hidden_matrix[0][0] = .8; weights_hidden_matrix[0][1] = .4; weights_hidden_matrix[0][2] = .3
weights_hidden_matrix[1][0] = .2; weights_hidden_matrix[1][1] = .9; weights_hidden_matrix[1][2] = .5

#In1    [I1 I2]   N1 N2 N3           H1            H2           H3
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
I7 = input_matrix[3][0]
I8 = input_matrix[3][1]
H1sum = I7 * weights_hidden_matrix[0][0] + I8 * weights_hidden_matrix[1][0]
H2sum = I7 * weights_hidden_matrix[0][1] + I8 * weights_hidden_matrix[1][1]
H3sum = I7 * weights_hidden_matrix[0][2] + I8 * weights_hidden_matrix[1][2]
H1 = sigmoid(H1sum)
H2 = sigmoid(H2sum)
H3 = sigmoid(H3sum)
print 'Hidden Values'
print H1
print H2
print H3
print
Osum = H1 * weights_output_matrix[0][0] + H2 * weights_output_matrix[1][0] + H3 * weights_output_matrix[2][0]
output = sigmoid(Osum)
print 'Output Value'
print output
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
print 'DeltaOutputSum:'
print DeltaOutputSum
print

delta_weights_output = weights_output_matrix.copy()
updated_weights_output = weights_output_matrix.copy()
delta_weights_output[0][0] = DeltaOutputSum / H1;
delta_weights_output[1][0] = DeltaOutputSum / H2;
delta_weights_output[2][0] = DeltaOutputSum / H3;

updated_weights_output[0][0] = weights_output_matrix[0][0] + delta_weights_output[0][0]
updated_weights_output[1][0] = weights_output_matrix[1][0] + delta_weights_output[1][0]
updated_weights_output[2][0] = weights_output_matrix[2][0] + delta_weights_output[1][0]

print 'DeltaOutputWeights:'
print delta_weights_output
print
print 'OldOutputWeights:'
print weights_output_matrix
print
print 'NewOutputWeights:'
print updated_weights_output
print

#                  ( How much each hidden layer affects DeltaOutputSum) * rate of change of input of H
H1delta_hidden_sum = DeltaOutputSum / weights_output_matrix[0][0] * sigmoid_prime(H1sum)
H2delta_hidden_sum = DeltaOutputSum / weights_output_matrix[1][0] * sigmoid_prime(H2sum)
H3delta_hidden_sum = DeltaOutputSum / weights_output_matrix[2][0] * sigmoid_prime(H3sum)


print 'DeltaHiddenSum:'
print H1delta_hidden_sum
print H2delta_hidden_sum
print H3delta_hidden_sum
print


delta_weights_input = weights_hidden_matrix.copy()
new_weights_input = weights_hidden_matrix.copy()
# [Weight into H1, Weight into H2, Weight into H3]
# [Weight into H1, Weight into H2, Weight into H3]
delta_weights_input[0][0] = H1delta_hidden_sum / I7; delta_weights_input[0][1] = H2delta_hidden_sum / I7; delta_weights_input[0][2] = H3delta_hidden_sum / I7
delta_weights_input[1][0] = H1delta_hidden_sum / I8; delta_weights_input[1][1] = H2delta_hidden_sum / I8; delta_weights_input[1][2] = H3delta_hidden_sum / I8

new_weights_input[0][0] += delta_weights_input[0][0]; new_weights_input[0][1] += delta_weights_input[0][1]; new_weights_input[0][2] += delta_weights_input[0][2]
new_weights_input[1][0] += delta_weights_input[1][0]; new_weights_input[1][1] += delta_weights_input[1][1]; new_weights_input[1][2] += delta_weights_input[1][2]

print 'DeltaInputWeights:'
print delta_weights_input
print
print 'OldInputWeights:'
print weights_hidden_matrix
print
print 'NewInputWeights:'
print new_weights_input
print



#Forward Propagation starts
print '############################################'
print '### Forward Propagation for input (1, 1) ###'
print '############################################'
print
I7 = input_matrix[3][0]
I8 = input_matrix[3][1]
H1sum = I7 * new_weights_input[0][0] + I8 * new_weights_input[1][0]
H2sum = I7 * new_weights_input[0][1] + I8 * new_weights_input[1][1]
H3sum = I7 * new_weights_input[0][2] + I8 * new_weights_input[1][2]
H1 = sigmoid(H1sum)
H2 = sigmoid(H2sum)
H3 = sigmoid(H3sum)
print 'Hidden Values'
print H1
print H2
print H3
print
Osum = H1 * updated_weights_output[0][0] + H2 * updated_weights_output[1][0] + H3 * updated_weights_output[2][0]
output = sigmoid(Osum)
print 'Output Value'
print output
print














"""
Notes

Reference:
http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/

H1 * W1

H2 * W2

H3 * W3

----------------------


OutputSum = (H1*W1) + (H2*W2) + (H3*W3)

Target = 0
Calculated = 0.77
Target - calculated = -0.77

Delta output sum = S'(sum) * (output sum margin of error)
Delta output sum = S'(1.235) * (-0.77)
Delta output sum = -0.13439890643886018
//S'(sum) : slope or rate of change at our output sum.
//Multiple the rate of change by our delta error(target result - calculated result).
//This gives us our delta output sum.
#Since the output sum margin of error is the difference in the result,
#we can simply multiply that with the rate of change to give us the delta output sum

DeltaOutputSum / H1 = How much W1 is responsible for of the delta
DeltaOutputSum / H2 = How much W2 is responsible for of the delta
DeltaOutputSum / H3 = How much W3 is responsible for of the delta
//Now we distrubute out much each hidden neuron is responsible for
//the delta output sum

hidden result 1 = 0.73105857863
hidden result 2 = 0.78583498304
hidden result 3 = 0.68997448112

Delta weights = delta output sum / hidden layer results
Delta weights = -0.1344 / [0.73105, 0.78583, 0.69997]
Delta weights = [-0.1838, -0.1710, -0.1920]

old w7 = 0.3
old w8 = 0.5
old w9 = 0.9

new w7 = 0.1162
new w8 = 0.329
new w9 = 0.708


----------------------------------------------
// Scaling to 1 helps make sense of distributing the
// DeltaOutputSum amoung the hidden neurons to calculate
// new weights.
Output 1 = .5*W1 + .25*W2 + .25*W3

DeltaOutput (-.1) =

-.1/.5
----------------------------------------------
----------------------------------------------
----------------------------------------------
----------------------------------------------


 1: Delta hidden sum = delta output sum / hidden-to-outer weights * S'(hidden sum)
*2: Delta hidden sum = -0.1344 / [0.3, 0.5, 0.9] * S'([1, 1.3, 0.8])
 3: Delta hidden sum = [-0.448, -0.2688, -0.1493] * [0.1966, 0.1683, 0.2139]
 4: Delta hidden sum = [-0.088, -0.0452, -0.0319]


*2:  DeltaOutput / W1 = How much H1 is responsible for of the delta * the slope(rate of change of the hidden sums)


----------------------------------------------
----------------------------------------------
----------------------------------------------
----------------------------------------------


input 1 = 1
input 2 = 1

Delta weights = delta hidden sum / input data
Delta weights = [-0.088, -0.0452, -0.0319] / [1, 1]
Delta weights = [-0.088, -0.0452, -0.0319, -0.088, -0.0452, -0.0319]

old w1 = 0.8
old w2 = 0.4
old w3 = 0.3
old w4 = 0.2
old w5 = 0.9
old w6 = 0.5

new w1 = 0.712
new w2 = 0.3548
new w3 = 0.2681
new w4 = 0.112
new w5 = 0.8548
new w6 = 0.4681

"""



