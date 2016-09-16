#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <iostream>
#include <iomanip>

#include <queue>
#include <vector>

class data;
class node;
class synapse;

class node {
public:
	uint64_t identification;

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
	uint64_t identification;

	bool ready;
	float output;
	float weight;

	std::vector<float> delta_weights;
	class node *forward_node;
	class node *reverse_node;
};

class data {
public:
	std::vector<float> inputs;
	std::vector<float> outputs;
};

class Network {
private:
	static const float learn_rate;
	uint64_t num_nodes;
	uint64_t num_synapses;
	std::queue<class node *> processing_queue;

	class node* create_node(uint64_t identification, float value);
	void connect_nodes(class node *from_node, class node *to_node,
		uint64_t identification, float weight);

	float get_initial_weight();
	void nodes_reset();
	void synapses_reset();
	float sigmoid(float input);
	float sigmoid_prime(float input);
	void set_inputs_outputs(const class data &data);
	void update_weights();

public:
	std::vector<class data> *training_data;
	std::vector<class node *> input_nodes;
	std::vector<class node *> hidden_nodes;
	std::vector<class node *> output_nodes;
	std::vector<class node *> nodes;
	std::vector<class synapse *> synapses;

	Network(std::vector<class data> *training_data, int num_hidden);
	void create_connections_default();
	void create_connections(int layers);

	void train(int num_epochs);
	void process(std::vector<class data> *data);
	void print_results();
	void forward_propagate();
	void back_propagate();
};

#endif //_NETWORK_H_
