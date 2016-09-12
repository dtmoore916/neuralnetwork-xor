#include "simple.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <queue>
#include <vector>

std::vector<struct node *> gNodes;
std::vector<struct synapse *> gSynapses;

std::vector<struct node *> gInputNodes;
std::vector<struct node *> gOutputNodes;
std::queue<struct node *> gProcessingQueue;


float sigmoid(float input)
{
	return 1/(1+exp(-input));
}

float sigmoid_prime(float input)
{
	return sigmoid(input) * (1-sigmoid(input));
}

struct node* create_node(std::string name, float value)
{
	struct node *node = (struct node*)calloc(1, sizeof(struct node));
	gNodes.push_back(node);
	node->name = name;
	node->value = value;
	return node;
}

void connect_nodes(struct node *from_node, struct node *to_node, float weight)
{
	struct synapse *synapse;

	synapse = (struct synapse *)calloc(1, sizeof(*synapse));
	synapse->name = from_node->name + " -> " + to_node->name;
	synapse->weight = weight;
	synapse->forward_node = to_node;
	synapse->reverse_node = from_node;

	from_node->synapse_outputs.push_back(synapse);
	to_node->synapse_inputs.push_back(synapse);
	gSynapses.push_back(synapse);
}

void forward_propagate()
{
	for(int i = 0; i < gInputNodes.size(); ++i) {
		gProcessingQueue.push(gInputNodes[i]);
	}

	// reset all synapses
	for(int i = 0; i < gSynapses.size(); ++i) {
		gSynapses[i]->ready = false;
	}

	while(gProcessingQueue.size() != 0) {
		bool node_ready = true;
		struct node *node = gProcessingQueue.front();
		gProcessingQueue.pop();

		// check all nodes input synapses.
		// (Inputs will succeed here. size() == 0)
		for(int i = 0; i < node->synapse_inputs.size(); ++i) {
			if(!node->synapse_inputs[i]->ready) {
				node_ready = false;
			}
		}

		if (node_ready) {
			if (node->synapse_inputs.size() == 0) {
				//input node
				node->activated_value = node->value;
			} else {
				for(int i = 0; i < node->synapse_inputs.size(); ++i) {
					node->value += node->synapse_inputs[i]->output;
				}
				node->activated_value = sigmoid(node->value);

				#if 1
				printf("%s: value: %f, activated: %f\n",
					node->name.c_str(), node->value,
					node->activated_value);
				#endif
			}

			// process my output synapse
			for(int i = 0; i < node->synapse_outputs.size(); ++i) {
				node->synapse_outputs[i]->output = node->synapse_outputs[i]->weight *
					node->activated_value;
				node->synapse_outputs[i]->ready = true;

				// check if the node connected to my output is ready to be
				// processed
				struct node *next_node = node->synapse_outputs[i]->forward_node;
				bool next_node_ready = true;
				if (next_node->synapse_inputs.size() > 0) {
					for(int j = 0; j < next_node->synapse_inputs.size(); ++j) {
						if (!next_node->synapse_inputs[j]->ready) {
							next_node_ready = false;
						}
					}

					if (next_node_ready) {
						gProcessingQueue.push(next_node);
					}
				}
			}
		}
	}
}

void back_propagate()
{

	//Delta output sum = S'(1.235) * (-0.77)
	//DOS = how much node value must change by
	// divide that amound the synapes weights

	// DOS / Original Weights * S'(hidden->value) : gives how much hidden value should change
	// distribute this amount the inputs. (divide)
	//Delta hidden sum = -0.1344 / [0.3, 0.5, 0.9] * S'([1, 1.3, 0.8])


	// For each output node calculate delta error
	// DOS =  S'(output) * dError
	// For synapses connected to output node distribute the DOS

	// Synapses connected to previous node:
	// DHS = DOS / original synapse weight * S'(previous nodes output)


	// commonized
	// Error for the node distribute amoung synapses
	// Error for the node: S'(value) * dError
	// dError for output is expected - activated_value
	// dError for previous nodes:
	// summation of:
	//     next node's: Delta Sum Output / original synapse weight


	// Outputs special case just calculate dError
	// Inputs should just flow through no inputs synapses to weight

	// reset all nodes
	for(int i = 0; i < gNodes.size(); ++i) {
		gNodes[i]->ready = false;
	}

	for(int i = 0; i < gOutputNodes.size(); ++i) {
		gProcessingQueue.push(gOutputNodes[i]);
	}

	while(gProcessingQueue.size() != 0) {
		struct node *node = gProcessingQueue.front();
		gProcessingQueue.pop();

		if (node->synapse_inputs.size() == 0)
			continue; // Input node

		if (node->synapse_outputs.size() == 0) {
			/*** Output node ***/
			float delta_error = node->target_value - node->activated_value;

			node->delta_output_sum = sigmoid_prime(node->value) * (delta_error);

			for(int i = 0; i < node->synapse_inputs.size(); ++i) {
				struct synapse *synapse = node->synapse_inputs[i];
				struct node *prev_node = synapse->reverse_node;

				synapse->updated_weight = synapse->weight +
					(node->delta_output_sum / prev_node->activated_value); // TODO: * learn rate

				#if 1
				printf("%s: DOS: %f weight: %f, updated_weight: %f\n",
					synapse->name.c_str(), node->delta_output_sum, synapse->weight,
					synapse->updated_weight);
				#endif

				// Just add my input nodes to queue, TODO: only add if not already queued
				if (synapse->reverse_node != NULL) {
					gProcessingQueue.push(synapse->reverse_node);
				}
			}

			node->ready = true;
		} else {
			// Hidden node
			float delta_output_sum_total = 0.0;

			for(int i = 0; i < node->synapse_outputs.size(); ++i) {
				struct node *next_node = node->synapse_outputs[i]->forward_node;

				if (!next_node->ready) {
					// This node is not ready yet.  It will be added
					// again by one of its outputs.
					continue;
				}

				delta_output_sum_total +=
					(next_node->delta_output_sum / node->synapse_outputs[i]->weight)
					* sigmoid_prime(node->value);
			}

			node->delta_output_sum = delta_output_sum_total / node->synapse_outputs.size();

			for(int i = 0; i < node->synapse_inputs.size(); ++i) {
				struct synapse *synapse = node->synapse_inputs[i];
				struct node *prev_node = synapse->reverse_node;

				synapse->updated_weight = synapse->weight +
					(node->delta_output_sum / prev_node->activated_value); // TODO: * learn rate

				#if 1
				printf("%s: DOS: %f activated_value: %f weight: %f, updated_weight: %f\n",
					synapse->name.c_str(), node->delta_output_sum, node->activated_value,
					synapse->weight, synapse->updated_weight);
				#endif

				// Just add my input nodes to queue, TODO: only add if not already queued
				if (synapse->reverse_node != NULL) {
					gProcessingQueue.push(synapse->reverse_node);
				}
			}

			node->ready = true;
		}
	}

	for(int i = 0; i < gSynapses.size(); ++i) {
		gSynapses[i]->weight = gSynapses[i]->updated_weight;
	}
}


int main(int argc, char *argv[])
{
	printf("Program Started\n");

	struct node *input1 = create_node("I1", 1.0);
	struct node *input2 = create_node("I2", 1.0);

	struct node *hidden1 = create_node("H1", 0.0);
	struct node *hidden2 = create_node("H2", 0.0);
	struct node *hidden3 = create_node("H3", 0.0);

	struct node *output1 = create_node("O1", 0.0);

	connect_nodes(input1, hidden1, 0.8);
	connect_nodes(input1, hidden2, 0.4);
	connect_nodes(input1, hidden3, 0.3);

	connect_nodes(input2, hidden1, 0.2);
	connect_nodes(input2, hidden2, 0.9);
	connect_nodes(input2, hidden3, 0.5);

	connect_nodes(hidden1, output1, 0.3);
	connect_nodes(hidden2, output1, 0.5);
	connect_nodes(hidden3, output1, 0.9);

	gInputNodes.push_back(input1);
	gInputNodes.push_back(input2);
	gOutputNodes.push_back(output1);

	forward_propagate();

	printf("Output Value: %f\n", output1->activated_value);

	back_propagate();


	for(int i = 0; i < gNodes.size(); i++) {
		free(gNodes[i]);
	}

	for(int i = 0; i < gSynapses.size(); i++) {
		free(gSynapses[i]);
	}

	printf("Program Finished\n");
	return 0;
}

