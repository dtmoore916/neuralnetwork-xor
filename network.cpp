#include "network.h"

const float Network::learn_rate = 1.0;

Network::Network(std::vector<class data> *training_data, int num_hidden)
{
	// Initialize random
	srand(1.0);  // srand (time(NULL));

	this->training_data = training_data;

	for(int i = 0; i < (*training_data)[0].inputs.size(); ++i) {
		class node *node = create_node("I", 0);

		nodes.push_back(node);
		input_nodes.push_back(node);
	}

	for(int i = 0; i < num_hidden; ++i) {
		class node *node = create_node("H", 0);

		nodes.push_back(node);
		hidden_nodes.push_back(node);
	}

	for(int i = 0; i < (*training_data)[0].outputs.size(); ++i) {
		class node *node = create_node("O", 0);

		nodes.push_back(node);
		output_nodes.push_back(node);
	}
}

void Network::create_connections_default()
{
	for(int i = 0; i < input_nodes.size(); ++i) {
		for(int j = 0; j < hidden_nodes.size(); ++j) {
			connect_nodes(input_nodes[i], hidden_nodes[j], get_initial_weight());
		}
	}

	for(int i = 0; i < hidden_nodes.size(); ++i) {
		for(int j = 0; j < output_nodes.size(); ++j) {
			connect_nodes(hidden_nodes[i], output_nodes[j], get_initial_weight());
		}
	}
}

void Network::create_connections(int layers)
{

}

class node* Network::create_node(std::string name, float value)
{
	class node *node = (class node*)calloc(1, sizeof(class node));
	node->name = name;
	node->value = value;
	return node;
}

void Network::connect_nodes(class node *from_node, class node *to_node, float weight)
{
	class synapse *synapse;

	synapse = (class synapse *)calloc(1, sizeof(*synapse));
	synapse->name = from_node->name + " -> " + to_node->name;
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
		synapses[i]->output = 0;
	}
}

void Network::set_inputs_outputs(const class data &data)
{
	for(int i = 0; i < data.inputs.size() && i < input_nodes.size(); ++i) {
		input_nodes[i]->value = data.inputs[i];
		if (input_nodes[i]->value == 0.0)
			input_nodes[i]->value = 0.01;
	}

	for(int i = 0; i < data.inputs.size() && i < output_nodes.size(); ++i) {
		output_nodes[i]->target_value = data.outputs[i];
		if (output_nodes[i]->target_value == 0.0)
			output_nodes[i]->target_value = 0.01;
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

void Network::forward_propagate()
{
	for(int i = 0; i < input_nodes.size(); ++i) {
		processing_queue.push(input_nodes[i]);
	}

	nodes_reset();
	synapses_reset();

	while(processing_queue.size() != 0) {
		class node *node = processing_queue.front();
		bool node_ready = true;

		processing_queue.pop();

		// check all nodes input synapses.
		// (Inputs will succeed here. size() == 0)
		for(int i = 0; i < node->synapse_inputs.size(); ++i) {
			if(!node->synapse_inputs[i]->ready) {
				node_ready = false;
			}
		}

		if (!node_ready)
			continue;

		if (node->synapse_inputs.size() == 0) {
			//input node
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

			// check if the node connected to my output is ready to be
			// processed
			class node *next_node = node->synapse_outputs[i]->forward_node;
			bool next_node_ready = true;
			if (next_node->synapse_inputs.size() > 0) {
				for(int j = 0; j < next_node->synapse_inputs.size(); ++j) {
					if (!next_node->synapse_inputs[j]->ready) {
						next_node_ready = false;
					}
				}

				if (next_node_ready) {
					processing_queue.push(next_node);
				}
			}
		}
	}
}

void Network::back_propagate()
{
	for(int i = 0; i < output_nodes.size(); ++i) {
		processing_queue.push(output_nodes[i]);
	}

	nodes_reset();

	while(processing_queue.size() != 0) {
		class node *node = processing_queue.front();

		processing_queue.pop();

		if (node->synapse_inputs.size() == 0)
			continue; // Input nodes are not processed

		bool node_ready = true;
		for(int i = 0; i < node->synapse_outputs.size(); ++i) {
			class node *next_node = node->synapse_outputs[i]->forward_node;
			if (!next_node->ready) {
				node_ready = false;
			}
		}

		if (!node_ready) {
			// This node is not ready yet.  It will be added
			// again by one of its outputs.
			continue;
		}

		/* Only Hidden nodes will loop through here.  Output Nodes have
		 * no output synapses */
		float delta_output_sum_total = 0.0;
		for(int i = 0; i < node->synapse_outputs.size(); ++i) {
			class node *next_node = node->synapse_outputs[i]->forward_node;

			delta_output_sum_total +=
				sigmoid_prime(node->value) *
				(next_node->delta_output_sum * node->synapse_outputs[i]->weight);
		}

		// Now compute this nodes delta output sum
		if (node->synapse_outputs.size() != 0) {
			// Hidden node
			node->delta_output_sum = (delta_output_sum_total /
				node->synapse_outputs.size());
		} else {
			// Output node
			float delta_error = node->target_value - node->activated_value;

			node->delta_output_sum = sigmoid_prime(node->value) * (delta_error);
		}

		// Now distribute the delta output sum amoung my input synapses
		for(int i = 0; i < node->synapse_inputs.size(); ++i) {
			class synapse *synapse = node->synapse_inputs[i];
			class node *prev_node = synapse->reverse_node;

			synapse->delta_weights.push_back(
				(node->delta_output_sum * prev_node->activated_value)
				* learn_rate);

			// Just add my input nodes to queue.
			if (synapse->reverse_node != NULL) {
				processing_queue.push(synapse->reverse_node);
			}
		}

		node->ready = true;
	}
}

