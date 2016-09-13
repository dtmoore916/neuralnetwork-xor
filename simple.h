#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <vector>
#include <string>

struct node;
struct synapse;

struct node {
	std::string name;

	bool ready;
	float value;
	float activated_value;
	float target_value;
	float delta_output_sum;

	std::vector<struct synapse *> synapse_inputs;
	std::vector<struct synapse *> synapse_outputs;
};

struct synapse {
	std::string name;

	bool ready;
	float output;
	float weight;
	float updated_weight;

	std::vector<float> updated_weights;
	struct node *forward_node;
	struct node *reverse_node;
};

#endif// _SIMPLE_H_
