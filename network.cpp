#include "network.h"

const float Network::learn_rate = 1.0;

Network::Network(std::vector<class data> *training_data, int num_hidden)
{
	// Initialize random
	srand(1.0);  // srand (time(NULL));

	this->num_nodes = 0;
	this->num_synapses = 0;
	this->training_data = training_data;

	for(int i = 0; i < (*training_data)[0].inputs.size(); ++i) {
		class node *node = create_node(num_nodes++, 0);

		nodes.push_back(node);
		input_nodes.push_back(node);
	}

	for(int i = 0; i < num_hidden; ++i) {
		class node *node = create_node(num_nodes++, 0);

		nodes.push_back(node);
		hidden_nodes.push_back(node);
	}

	for(int i = 0; i < (*training_data)[0].outputs.size(); ++i) {
		class node *node = create_node(num_nodes++, 0);

		nodes.push_back(node);
		output_nodes.push_back(node);
	}
}

void Network::create_connections_default()
{
	for(int i = 0; i < input_nodes.size(); ++i) {
		for(int j = 0; j < hidden_nodes.size(); ++j) {
			connect_nodes(input_nodes[i], hidden_nodes[j],
			              num_synapses++, get_initial_weight());
		}
	}

	for(int i = 0; i < hidden_nodes.size(); ++i) {
		for(int j = 0; j < output_nodes.size(); ++j) {
			connect_nodes(hidden_nodes[i], output_nodes[j],
			              num_synapses++, get_initial_weight());
		}
	}
}

void Network::create_connections(int layers)
{

}

class node* Network::create_node(uint64_t identification, float value)
{
	class node *node = (class node*)calloc(1, sizeof(class node));
	node->identification = identification;
	node->value = value;
	return node;
}

void Network::connect_nodes(class node *from_node, class node *to_node,
                            uint64_t identification, float weight)
{
	class synapse *synapse;

	synapse = (class synapse *)calloc(1, sizeof(*synapse));
	synapse->identification = identification;
	synapse->weight = weight;
	synapse->forward_node = to_node;
	synapse->reverse_node = from_node;

	from_node->synapse_outputs.push_back(synapse);
	to_node->synapse_inputs.push_back(synapse);
	synapses.push_back(synapse);
}

void Network::print_results()
{
	float percent_error = 0.0;

	for(int i = 0; i < (*training_data).size(); ++i) {
		set_inputs_outputs((*training_data)[i]);
		forward_propagate();

		std::cout << std::setprecision(2) << std::fixed;
		std::cout << "[";

		for (int j = 0; j < input_nodes.size(); ++j) {
			if (j != 0)
				std::cout << ", ";
			std::cout << input_nodes[j]->value;
		}
		std::cout << "] ==> ";

		percent_error = 0.0;

		std::cout << std::setprecision(4) << std::fixed;
		std::cout << "[";

		for (int j = 0; j < output_nodes.size(); ++j) {
			if (j != 0)
				std::cout << ", ";
			std::cout << output_nodes[j]->activated_value;

			percent_error += fabs(
			                     (output_nodes[j]->target_value - output_nodes[j]->activated_value));
		}
		percent_error = percent_error / output_nodes.size() * 100.0;
		std::cout << std::setprecision(2) << std::fixed;
		std::cout << "] Error " << percent_error << "%";

		std::cout << std::endl;
	}
}

void Network::train(int num_epochs)
{
	for(int i = 0; i < num_epochs; ++i) {
		for(int j = 0; j < (*training_data).size(); ++j) {
			set_inputs_outputs((*training_data)[j]);
			forward_propagate();
			back_propagate();
		}
		update_weights();
	}
}

void Network::process(std::vector<class data> *data)
{
	for(int j = 0; j < (*data).size(); ++j) {
		set_inputs_outputs((*data)[j]);
		forward_propagate();
	}
}

float Network::get_initial_weight()
{
	static const int initial_weight_resolution = 10;

	/* generate secret number between 1 and initial_weight_resolution: */
	return 1.0 / (rand() % initial_weight_resolution + 1);
}

float Network::sigmoid(float input)
{
	return (1 / (1 + exp(-input)));
}

float Network::sigmoid_prime(float input)
{
	return sigmoid(input) * (1 - sigmoid(input));
}

void Network::nodes_reset(void)
{
	for(int i = 0; i < nodes.size(); ++i) {
		nodes[i]->ready = false;
	}
}

void Network::synapses_reset(void)
{
	for(int i = 0; i < synapses.size(); ++i) {
		synapses[i]->ready = false;
	}
}

void Network::set_inputs_outputs(const class data &data)
{
	for(int i = 0; i < data.inputs.size() && i < input_nodes.size(); ++i) {
		input_nodes[i]->value = data.inputs[i];
	}

	for(int i = 0; i < data.inputs.size() && i < output_nodes.size(); ++i) {
		output_nodes[i]->target_value = data.outputs[i];
	}
}

void Network::update_weights()
{
	for(int i = 0; i < synapses.size(); ++i) {
		for(int j = 0; j < synapses[i]->delta_weights.size(); ++j) {
			synapses[i]->weight += synapses[i]->delta_weights[j];
		}
		synapses[i]->delta_weights.clear();
	}
}

bool Network::inputs_ready(class node *node)
{
	bool node_ready = true;

	for(int i = 0; i < node->synapse_inputs.size(); ++i) {
		if(!node->synapse_inputs[i]->ready) {
			node_ready = false;
			break;
		}
	}

	return node_ready;
}

bool Network::outputs_ready(class node *node)
{
	bool node_ready = true;

	for(int i = 0; i < node->synapse_outputs.size(); ++i) {
		class node *next_node = node->synapse_outputs[i]->forward_node;

		if (!next_node->ready) {
			node_ready = false;
			break;
		}
	}

	return node_ready;
}

bool Network::is_input(class node *node)
{
	return node->synapse_inputs.size() == 0;
}

bool Network::is_output(class node *node)
{
	return node->synapse_outputs.size() == 0;
}


void Network::reset_network()
{
	std::queue<class node *> processing_queue;

	for(int i = 0; i < input_nodes.size(); ++i) {
		processing_queue.push(input_nodes[i]);
	}

	while(processing_queue.size() != 0) {
		class node *node = processing_queue.front();
		processing_queue.pop();

		node->ready = false;

		for(int i = 0; i < node->synapse_outputs.size(); ++i) {
			node->synapse_outputs[i]->ready = false;
			processing_queue.push(node->synapse_outputs[i]->forward_node);
		}
	}
}

void Network::forward_propagate()
{
	std::queue<class node *> processing_queue;

	for(int i = 0; i < input_nodes.size(); ++i) {
		processing_queue.push(input_nodes[i]);
	}

	reset_network();

	while(processing_queue.size() != 0) {
		class node *node = processing_queue.front();
		processing_queue.pop();

		if (node == NULL)
			continue;

		if (!inputs_ready(node))
			continue;

		if (is_input(node)) {
			node->activated_value = node->value;
		} else {
			node->value = 0;
			for(int i = 0; i < node->synapse_inputs.size(); ++i) {
				node->value += node->synapse_inputs[i]->output;
			}
			node->activated_value = sigmoid(node->value);
		}

		// process my output synapse
		for(int i = 0; i < node->synapse_outputs.size(); ++i) {
			node->synapse_outputs[i]->output = node->synapse_outputs[i]->weight *
			                                   node->activated_value;
			node->synapse_outputs[i]->ready = true;

			processing_queue.push(node->synapse_outputs[i]->forward_node);
		}
	}
}

void Network::back_propagate()
{
	std::queue<class node *> processing_queue;

	for(int i = 0; i < output_nodes.size(); ++i) {
		processing_queue.push(output_nodes[i]);
	}

	reset_network();

	while(processing_queue.size() != 0) {
		class node *node = processing_queue.front();
		processing_queue.pop();

		if (node == NULL)
			continue;

		if (is_input(node))
			continue; // No need to process Input nodes

		if (!outputs_ready(node))
			continue;

		if (is_output(node)) {
			float delta_error = node->target_value - node->activated_value;

			node->delta_output_sum = sigmoid_prime(node->value) * (delta_error);
		} else {
			float delta_output_sum_total = 0.0;

			for(int i = 0; i < node->synapse_outputs.size(); ++i) {
				class node *next_node = node->synapse_outputs[i]->forward_node;

				delta_output_sum_total +=
				    sigmoid_prime(node->value) *
				    (next_node->delta_output_sum * node->synapse_outputs[i]->weight);
			}

			node->delta_output_sum = (delta_output_sum_total /
			                          node->synapse_outputs.size());
		}

		// Now distribute the delta output sum amoung my input synapses
		for(int i = 0; i < node->synapse_inputs.size(); ++i) {
			class synapse *synapse = node->synapse_inputs[i];
			class node *prev_node = synapse->reverse_node;

			synapse->delta_weights.push_back(
			    (node->delta_output_sum * prev_node->activated_value)
			    * learn_rate);

			processing_queue.push(prev_node);
		}

		node->ready = true;
	}
}

