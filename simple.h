#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <vector>
#include <string>

struct node;
struct synapse;

struct node {
	std::string name;
	float value;
	float activated_value;
	float target_value;
	float delta_output_sum;
	bool ready;
	std::vector<struct synapse *> synapse_inputs;
	std::vector<struct synapse *> synapse_outputs;
};

struct synapse {
	std::string name;
	float weight;
	float updated_weight;
	float output;
	bool ready;
	struct node *forward_node;
	struct node *reverse_node;
};

#endif// _SIMPLE_H_
