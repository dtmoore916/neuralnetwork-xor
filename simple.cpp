#include "simple.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <queue>
#include <vector>

//#define DEBUG

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

			#ifdef DEBUG
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

void back_propagate()
{
	// For each node we need to compute it's delta output sum and distribute
	// that amoungst it's input synapses to find their new weight.
	// For a output node the DOS is delta_error * sigmoid_prime(node->value)
	// For a hidden node, we need to sum all the DOS from all output synapses.
	// the DOS of a hidden node is the next node down that (synapse(DOS) / synapse
	// weight original) * sigmoid_prime(node->value).

	for(int i = 0; i < gOutputNodes.size(); ++i) {
		gProcessingQueue.push(gOutputNodes[i]);
	}

	nodes_reset();

	while(gProcessingQueue.size() != 0) {
		struct node *node = gProcessingQueue.front();
		float delta_output_sum_total = 0.0;
		bool node_ready = true;

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

			// Add up all the delta output sums of all our output synapses
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
			//printf("delta_error: %f dos: %f value: %f s': %f", delta_error, node->delta_output_sum,
			//	node->value, sigmoid_prime(node->value));
		}

		float total_synapse_weight = 0.0;
		for(int i = 0; i < node->synapse_inputs.size(); ++i) {
			total_synapse_weight += node->synapse_inputs[i]->weight;
		}

		// Now distribute the delta output sum amoung my input synapses
		for(int i = 0; i < node->synapse_inputs.size(); ++i) {
			struct synapse *synapse = node->synapse_inputs[i];
			struct node *prev_node = synapse->reverse_node;

			//synapse->updated_weight = synapse->weight +
			//	(node->delta_output_sum * (synapse->weight / total_synapse_weight));

			//synapse->updated_weight = synapse->weight +
			//	((node->delta_output_sum / prev_node->activated_value) * 1.0/*learn rate*/);

			synapse->updated_weight = synapse->weight +
				((node->delta_output_sum * prev_node->activated_value) * 1.0/*learn rate*/);

			#ifdef DEBUG
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

	// actually update our weights the the newly calculated weights
	for(int i = 0; i < gSynapses.size(); ++i) {
		gSynapses[i]->updated_weights.push_back(gSynapses[i]->updated_weight);
	}
}

void update_weights(void)
{
	//printf("\nUpdate Weights:\n");
	for(int i = 0; i < gSynapses.size(); ++i) {
		gSynapses[i]->weight = 0;
		for(int j = 0; j < gSynapses[i]->updated_weights.size(); ++j) {
			gSynapses[i]->weight += gSynapses[i]->updated_weights[j];
			//printf("%s %f\n", gSynapses[i]->name.c_str(), gSynapses[i]->updated_weights[j]);
		}
		gSynapses[i]->weight /= gSynapses[i]->updated_weights.size();
		//printf("%s final %f\n", gSynapses[i]->name.c_str(), gSynapses[i]->weight);
		gSynapses[i]->updated_weights.clear();
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

	// Initial weights shouldn't mattter!!!...but they do.  Something is not working
	//connect_nodes(input1, hidden1, 0.8);
	//connect_nodes(input1, hidden2, 0.8);
	//connect_nodes(input1, hidden3, 0.8);

	//connect_nodes(input2, hidden1, 0.8);
	//connect_nodes(input2, hidden2, 0.8);
	//connect_nodes(input2, hidden3, 0.8);

	//connect_nodes(hidden1, output1, 0.8);
	//connect_nodes(hidden2, output1, 0.8);
	//connect_nodes(hidden3, output1, 0.8);



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

			//printf("Original:\n");
			//print_network();
			forward_propagate();
			//printf("\n\nForward:\n");
			//print_network();
			back_propagate();
			//printf("\n\nBackward:\n");
			//print_network();


			//printf("(%f, %f): %f  Target Value: %f Error: %f",
			//	input1->value, input2->value,
			//	output1->activated_value, output1->target_value,
			//	(output1->target_value - output1->activated_value)
			//	);
			//std::cin.ignore();
		}
		//printf("\n");

		update_weights();
	}

	printf("\n\n");
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

	printf("Program Finished\n");
	return 0;
}

