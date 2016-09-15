#include "network.h"

int main(int argc, char *argv[])
{
	class data data;
	std::vector<class data> training_data;

	data.inputs.push_back(1.0);
	data.inputs.push_back(1.0);  /* ==> */ data.outputs.push_back(0.0);
	training_data.push_back(data);

	data.inputs[0] = 0.0;
	data.inputs[1] = 1.0;  /* ==> */ data.outputs[0] = 1.0;
	training_data.push_back(data);

	data.inputs[0] = 0.0;
	data.inputs[1] = 0.0;  /* ==> */ data.outputs[0] = 0.0;
	training_data.push_back(data);

	data.inputs[0] = 1.0;
	data.inputs[1] = 0.0;  /* ==> */ data.outputs[0] = 1.0;
	training_data.push_back(data);

	Network network(&training_data, 3);
	network.create_connections_default();
	network.train(10000);
	network.print_results();

	return 0;
}

