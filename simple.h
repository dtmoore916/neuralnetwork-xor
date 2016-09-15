#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <iostream>

#include <queue>
#include <vector>

#if 0
class node;
class synapse;

class node {
public:
	std::string name;

	bool ready;
	float value;
	float activated_value;
	float target_value;
	float delta_output_sum;

	std::vector<class synapse *> synapse_inputs;
	std::vector<class synapse *> synapse_outputs;
};

class synapse {
public:
	std::string name;

	bool ready;
	float output;
	float weight;
	float updated_weight;

	std::vector<float> updated_weights;
	class node *forward_node;
	class node *reverse_node;
};
#endif

#endif// _SIMPLE_H_
