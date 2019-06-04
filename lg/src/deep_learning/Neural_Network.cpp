// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include "Neural_Network.hpp"
#include "../util/IOData.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	////////////////////////////////////////////////////////////
	Neural_Network::Neural_Network() {}

	////////////////////////////////////////////////////////////
	Neural_Network::Neural_Network(std::string filepath)
	{
		load(filepath);
	}

	////////////////////////////////////////////////////////////
	Neural_Network::~Neural_Network() { clear(); }

	////////////////////////////////////////////////////////////
	void Neural_Network::save(std::string filepath)
	{
		lg::IOData data("network");
		data.pushNode("node_count", (int)_nodes.size());
		data.pushNode("nodes");
		lg::IOData* node_data = data.findNode("nodes");
		ensure(node_data != NULL);
		for (int z = 0; z < (int)_nodes.size(); z++) {
			node_data->pushNode("node_" + std::to_string(z));
			lg::IOData* node = node_data->findNode("node_" + std::to_string(z));
			_nodes[z].save(*node);
		}
		if (!data.writeToFile(filepath)) {
			printf("Error in NeuralNework.cpp: can't save network to filepath %s\n", filepath.c_str());
		}
	}

	////////////////////////////////////////////////////////////
	void Neural_Network::load(std::string filepath)
	{
		lg::IOData data("");
		if (!data.loadFromFile(filepath)) {
			printf("Error in NeuralNework.cpp: can't load network from filepath %s\n", filepath.c_str());
			return;
		}

		clear();
		
		lg::IOData* node_count = data.findNode("node_count");
		ensure(node_count != NULL);
		lg::IOData* nodes = data.findNode("nodes");
		ensure(nodes != NULL);
		int node_count_val;
		node_count->get(node_count_val);
		for (int z = 0; z < (int)node_count_val; z++) {
			lg::IOData* node = nodes->findNode("node_" + std::to_string(z));
			_nodes.push_back(Node_Network(*node, this));
		}
	}

	////////////////////////////////////////////////////////////
	void Neural_Network::push(std::string node_name, std::string inputs_names, std::shared_ptr<Operation> operation)
	{
		std::vector<std::string> splitted_inputs_names = splitString(inputs_names, ',');
		_nodes.push_back(Node_Network(node_name, splitted_inputs_names, this, operation));
	}
	
	////////////////////////////////////////////////////////////
	Node_Network* Neural_Network::get_byname(std::string node_name)
	{
		for (int z = 0; z < (int)_nodes.size(); z++)
			if (_nodes[z].getName() == node_name)
				return &_nodes[z];
		return NULL;
	}
	
	////////////////////////////////////////////////////////////
	std::vector<std::string> Neural_Network::splitString(std::string s, char delimiter)
	{
		std::vector<std::string> substrings;

		if (s == "")
		{
			//Nothing to do
		}
		else
		{
			std::string block;
			std::stringstream stream(s);
			while(std::getline(stream, block, delimiter))
				substrings.push_back(block);
		}
		
		return substrings;
	}
	
	#ifdef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	void Neural_Network::run(CUDA_Tensor_float input, const bool training)
	{
		_nodes.front().setOperationOutput(input);

        //Feedforward
        for (int z = 0; z < (int)_nodes.size(); z++)
			_nodes[z].run(training);
	}
	
	////////////////////////////////////////////////////////////
	float Neural_Network::optimize(CUDA_Tensor_float input, CUDA_Tensor_float target, Optimizer* optimizer)
	{
		optimizer->fit(*this, input, target);
		return optimizer->getError();
	}
	
	////////////////////////////////////////////////////////////
	bool Neural_Network::test(CUDA_Tensor_float input, CUDA_Tensor_float target)
	{
		//Feedforward
		run(input, false);

		Tensor_float outputs_host = _nodes.back().getOperationOutput();
		Tensor_float target_host(target.size());
		target.copyToHost(target_host.pointer(), target_host.size());
		
		int target_id;
		target_host.max(NULL, &target_id);

		int output_id;
		outputs_host.max(NULL, &output_id);
		
		if (target_id == output_id) return true;
		else return false;
	}
	
	////////////////////////////////////////////////////////////
	CUDA_Tensor_float& Neural_Network::get_output(std::string node_name)
	{
		for (int z = 0; z < (int)_nodes.size(); z++)
			if (_nodes[z].getName() == node_name)
				return _nodes[z].getOperationOutputDevice();
		printf("Node with name  %s not found\n", node_name.c_str());
		exit(-1);
	}

	#else
	
	////////////////////////////////////////////////////////////
	void Neural_Network::run(Tensor_float input, const bool training)
	{
		_nodes.front().setOperationOutput(input);

		//Feedforward
		for (int z = 0; z < (int)_nodes.size(); z++)
			_nodes[z].run(training);
	}
	
	////////////////////////////////////////////////////////////
	float Neural_Network::optimize(Tensor_float input, Tensor_float target, Optimizer* optimizer)
	{
		optimizer->fit(*this, input, target);
		return optimizer->getError();
	}
	
	////////////////////////////////////////////////////////////
	Tensor_float& Neural_Network::get_output(std::string node_name)
	{
		for (int z = 0; z < (int)_nodes.size(); z++)
			if (_nodes[z].getName() == node_name)
				return _nodes[z].getOperationOutput();
		printf("Node with name  %s not found\n", node_name.c_str());
		exit(-1);
	}
	
	#endif
	
	////////////////////////////////////////////////////////////
	void Neural_Network::clear()
	{
		_nodes.clear();
	}

	////////////////////////////////////////////////////////////
	std::vector<Node_Network>& Neural_Network::getNodes()
	{
		return _nodes;
	}

	////////////////////////////////////////////////////////////
	void Neural_Network::printstack()
	{
		for (int z = 0; z < (int)_nodes.size(); z++)
			_nodes[z].print();
	}

} /* namespace lg */
