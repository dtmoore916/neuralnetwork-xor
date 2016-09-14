#include "simple.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

#include <queue>
#include <vector>

std::vector<struct node *> gNodes;
std::vector<struct synapse *> gSynapses;

std::vector<struct node *> gInputNodes;
std::vector<struct node *> gOutputNodes;

std::queue<struct node *> gProcessingQueue;

std::vector<float> gInput1_Values;
std::vector<float> gInput2_Values;
std::vector<float> gOutput_Values;

float divide(float dividend, float divisor)
{
	if (divisor == 0.0) return 0.0;
	return dividend / divisor;
}

float sigmoid(float input)
{
	return (1 / (1 + exp(-input)));
}

float sigmoid_prime(float input)
{
	return sigmoid(input) * (1 - sigmoid(input));
}

float get_random(int max)
{
	/* generate secret number between 1 and max: */
	return rand() % max + 1;
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

void print_network(void)
{
	printf("Nodes:\n");
	for(int i = 0; i < gNodes.size(); ++i) {
		printf("%s value %f activated %f target %f dos %f\n",
			gNodes[i]->name.c_str(), gNodes[i]->value,
			gNodes[i]->activated_value, gNodes[i]->target_value,
			gNodes[i]->delta_output_sum);
	}
	printf("\n");

	printf("Synapses:\n");
	for(int i = 0; i < gSynapses.size(); ++i) {
		printf("%s output %f weight %f updated_weight %f\n",
			gSynapses[i]->name.c_str(), gSynapses[i]->output,
			gSynapses[i]->weight,
			gSynapses[i]->updated_weight);
	}
	printf("\n");
}

void nodes_reset(void)
{
	for(int i = 0; i < gNodes.size(); ++i) {
		gNodes[i]->ready = false;
	}
}

void synapses_reset(void)
{
	for(int i = 0; i < gSynapses.size(); ++i) {
		gSynapses[i]->ready = false;
		gSynapses[i]->updated_weight = 0;
		gSynapses[i]->output = 0;
	}
}

void forward_propagate()
{
	for(int i = 0; i < gInputNodes.size(); ++i) {
		gProcessingQueue.push(gInputNodes[i]);
	}

	nodes_reset();
	synapses_reset();

	while(gProcessingQueue.size() != 0) {
		struct node *node = gProcessingQueue.front();
		bool node_ready = true;

		gProcessingQueue.pop();

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

void back_propagate()
{
	// For each node we need to compute it's delta output sum and distribute
	// that amoungst it's input synapses to find their new weight.
	// For a output node the DOS is delta_error * sigmoid_prime(node->value)
	// For a hidden node, we need to sum all the DOS from all output synapses.
	// the DOS of a hidden node is the next node down that (synapse(DOS) / synapse
	// weight original) * sigmoid_prime(node->value).

	// We start processing with the output nodes
	for(int i = 0; i < gOutputNodes.size(); ++i) {
		gProcessingQueue.push(gOutputNodes[i]);
	}

	nodes_reset();

	while(gProcessingQueue.size() != 0) {
		struct node *node;
		float delta_output_sum_total = 0.0;
		bool node_ready = true;

		node = gProcessingQueue.front();
		gProcessingQueue.pop();

		if (node->synapse_inputs.size() == 0)
			continue; // Input nodes are not processed

		for(int i = 0; i < node->synapse_outputs.size(); ++i) {
			struct node *next_node = node->synapse_outputs[i]->forward_node;
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
		delta_output_sum_total = 0;
		for(int i = 0; i < node->synapse_outputs.size(); ++i) {
			struct node *next_node = node->synapse_outputs[i]->forward_node;

			// Add up all the delta output sums of all our output synapses.
			// To distribute the next node's delta_output_sum back to each
			// of it's input synapses we need to just multiple the DOS by the
			// original weight.  Since we are dealing with numbers between 0 and 1
			// a smaller weight * DOS means this affects the output less.
			//   *I mistakenly was dividing here which does the oposite of what
			//    we want since dividing by a smaller number results in a greater
			//    in pact from a synapse that should actually have less of an impact.
			delta_output_sum_total +=
				(next_node->delta_output_sum * node->synapse_outputs[i]->weight)
				* sigmoid_prime(node->value);
		}

		// Now compute this nodes delta output sum
		if (node->synapse_outputs.size() != 0) {
			/*** Hidden node ***/
			node->delta_output_sum = (delta_output_sum_total /
				node->synapse_outputs.size());
		} else {
			/*** Output node ***/
			float delta_error = node->target_value - node->activated_value;

			node->delta_output_sum = sigmoid_prime(node->value) * (delta_error);
		}

		// Now distribute the delta output sum amoung my input synapses
		for(int i = 0; i < node->synapse_inputs.size(); ++i) {
			struct synapse *synapse = node->synapse_inputs[i];
			struct node *prev_node = synapse->reverse_node;

			//   *I mistakenly was dividing here(DOS) which does the oposite of what
			//    we want since dividing by a smaller number results in a greater
			//    in pact from a synapse that should actually have less of an impact.
			synapse->updated_weight = synapse->weight +
				((node->delta_output_sum * prev_node->activated_value) * 1.0/*learn rate*/);

			// Just add my input nodes to queue, TODO: only add if not already queued
			if (synapse->reverse_node != NULL) {
				gProcessingQueue.push(synapse->reverse_node);
			}
		}

		node->ready = true;
	}

	// actually update our weights the the newly calculated weights
	for(int i = 0; i < gSynapses.size(); ++i) {
		gSynapses[i]->updated_weights.push_back(gSynapses[i]->updated_weight);
	}
}

void update_weights(void)
{
	for(int i = 0; i < gSynapses.size(); ++i) {
		gSynapses[i]->weight = 0;
		for(int j = 0; j < gSynapses[i]->updated_weights.size(); ++j) {
			gSynapses[i]->weight += gSynapses[i]->updated_weights[j];
		}
		gSynapses[i]->weight /= gSynapses[i]->updated_weights.size();
		gSynapses[i]->updated_weights.clear();
	}
}

int main(int argc, char *argv[])
{
	struct node *input1 = create_node("I1", 1.0);
	struct node *input2 = create_node("I2", 1.0);

	struct node *hidden1 = create_node("H1", 0.0);
	struct node *hidden2 = create_node("H2", 0.0);
	struct node *hidden3 = create_node("H3", 0.0);

	struct node *output1 = create_node("O1", 0.0);

	// Initialize random seed: constant value for testing
	srand(1.0);  // srand (time(NULL));

	// Random initial weights in 0.1 increments
	connect_nodes(input1, hidden1, 1 / get_random(10));
	connect_nodes(input1, hidden2, 1 / get_random(10));
	connect_nodes(input1, hidden3, 1 / get_random(10));

	connect_nodes(input2, hidden1, 1 / get_random(10));
	connect_nodes(input2, hidden2, 1 / get_random(10));
	connect_nodes(input2, hidden3, 1 / get_random(10));

	connect_nodes(hidden1, output1, 1 / get_random(10));
	connect_nodes(hidden2, output1, 1 / get_random(10));
	connect_nodes(hidden3, output1, 1 / get_random(10));

	gInputNodes.push_back(input1);
	gInputNodes.push_back(input2);

	gOutputNodes.push_back(output1);

	gInput1_Values.push_back(1.0);
	gInput2_Values.push_back(1.0); /* --> */ gOutput_Values.push_back(0.01);

	gInput1_Values.push_back(0.01);
	gInput2_Values.push_back(1.0); /* --> */ gOutput_Values.push_back(1.0);

	gInput1_Values.push_back(0.01);
	gInput2_Values.push_back(0.01); /* --> */ gOutput_Values.push_back(0.01);

	gInput1_Values.push_back(1.0);
	gInput2_Values.push_back(0.01); /* --> */ gOutput_Values.push_back(1.0);

	int num_inputs_to_process = 4;

	for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < num_inputs_to_process; ++j) {
			input1->value = gInput1_Values[j];
			input2->value = gInput2_Values[j];
			output1->target_value = gOutput_Values[j];

			forward_propagate();
			back_propagate();
		}
		update_weights();
	}

	printf("\n Results:\n");
	for (int j = 0; j < num_inputs_to_process; j++) {
		input1->value = gInput1_Values[j];
		input2->value = gInput2_Values[j];
		output1->target_value = gOutput_Values[j];

		forward_propagate();
		printf("(%f, %f): %f  Target Value: %f\n",
			input1->value, input2->value,
			output1->activated_value, output1->target_value);
	}


	for(int i = 0; i < gNodes.size(); i++) {
		free(gNodes[i]);
	}

	for(int i = 0; i < gSynapses.size(); i++) {
		free(gSynapses[i]);
	}

	return 0;
}

