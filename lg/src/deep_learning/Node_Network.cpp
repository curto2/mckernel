// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Node_Network.hpp"
#include "Neural_Network.hpp"
#include "../util/ensure.hpp"
#include <stdlib.h>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	Node_Network::Node_Network(std::string node_name, std::vector<std::string> input_names,
		Neural_Network* network, std::shared_ptr<Operation> operation)
	{
		ensure_print(node_name != "", "Error, empty node name not allowed\n");

		_name = node_name;
		_input_names = input_names;
		_network = network;
		_operation = operation;
		
		checkInvalidInputNames();

		initInputsIndiciesVector();
		initInputsOperationsVector();
		
		ensure(_input_indicies.size() == _input_names.size());

		_operation->initialize(_input_operations);
	}
	
	////////////////////////////////////////////////////////////
	Node_Network::Node_Network(lg::IOData& data, Neural_Network* network)
	{
		_network = network;
		load(data);
		initInputsOperationsVector();
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::checkInvalidInputNames()
	{
		for (int k  = 0; k < (int)_input_names.size(); k++) {
			bool found = false;
			for (int z = 0; z < (int)_network->getNodes().size(); z++) {
				if (_network->getNodes()[z].getName() == _input_names[k]) {
					found = true;
					break;
				}
			}
			
			//Invalid name
			if (found == false) {
				printf("Error while creating node '%s', invalid input name %s\n", _name.c_str(), _input_names[k].c_str());
				exit(-1);
			}
		}
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::initInputsIndiciesVector()
	{
		_input_indicies = std::vector<int>();
		int index = 0;
		for (int z = 0; z < (int)_network->getNodes().size(); z++)
			for (int k  = 0; k < (int)_input_names.size(); k++)
				if (_network->getNodes()[z].getName() == _input_names[k]) {
					_input_indicies.push_back(z);
					index++;
				}
	}

	////////////////////////////////////////////////////////////
	void Node_Network::initInputsOperationsVector()
	{
		_input_operations = std::vector<Operation*>(_input_indicies.size());
		for (int z = 0; z < (int)_input_operations.size(); z++)
			_input_operations[z] = _network->getNodes()[_input_indicies[z]].getOperation();
	}

	////////////////////////////////////////////////////////////
	void Node_Network::load(lg::IOData& data)
	{
		lg::IOData* node_name = data.findNode("node_name");
		ensure(node_name != NULL);
		lg::IOData* inputs = data.findNode("inputs");
		ensure(inputs != NULL);
		lg::IOData* operation = data.findNode("operation");
		ensure(operation != NULL);
		node_name->get(_name);
		_input_names.clear();
		_input_indicies.clear();
		for (int z = 0; z < (int)inputs->getSubNodes().size(); z++) {		
			_input_names.push_back(inputs->getSubNodes()[z].getName());
			_input_indicies.push_back(0);
			inputs->getSubNodes()[z].get(_input_indicies.back());
		}
		_operation = Operation::loadFromFile(*operation);
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::save(lg::IOData& data)
	{
		data.pushNode("node_name", _name);
		data.pushNode("inputs");
		lg::IOData& inputs = *data.findNode("inputs");
		for (int z = 0; z < (int)_input_names.size(); z++)
			inputs.pushNode(_input_names[z], _input_indicies[z]);
		data.pushNode("operation");
		lg::IOData& operation_data = *data.findNode("operation");
		Operation::saveToFile(_operation, operation_data);
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::run(bool training)
	{
		_operation->run(_input_operations, training);
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::backprop()
	{
		_operation->backprop(_input_operations);
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::accumulate_deltas()
	{
		_operation->accumulate_deltas(_input_operations);
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::update_parameters(const float learningrate)
	{
		_operation->update_parameters(learningrate);
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::reset_errors()
	{
		_operation->reset_errors();
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::reset_deltas(const float momentum)
	{
		_operation->reset_deltas(momentum);
	}

	////////////////////////////////////////////////////////////
	const std::string Node_Network::getName() const
	{
		return _name;
	}

	////////////////////////////////////////////////////////////
	const std::vector<std::string> Node_Network::getInputsNames()
	{
		return _input_names;
	}
	
	////////////////////////////////////////////////////////////
	const std::vector<int> Node_Network::getInputsIndicies()
	{
		return _input_indicies;
	}
	
	////////////////////////////////////////////////////////////
	Operation* Node_Network::getOperation()
	{
		return _operation.get();
	}
			
	#ifdef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	Tensor_float Node_Network::getOperationOutput()
	{
		Tensor_float tmp(_operation->_outputs.width(), _operation->_outputs.height(), _operation->_outputs.depth());
		_operation->_outputs.copyToHost(&tmp[0], tmp.size());
		return tmp;
	}
	
	////////////////////////////////////////////////////////////
	const CUDA_Tensor_float& Node_Network::getOperationOutputDevice()
	{
		return _operation->_outputs;
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::setOperationOutput(CUDA_Tensor_float& output)
	{
		_operation->_outputs.point(output);	
	}

	#else
	
	////////////////////////////////////////////////////////////
	Tensor_float& Node_Network::getOperationOutput()
	{
		return _operation->_outputs;
	}
	
	////////////////////////////////////////////////////////////
	void Node_Network::setOperationOutput(Tensor_float& output)
	{
		_operation->_outputs.point(output);	
	}

	#endif
	
	////////////////////////////////////////////////////////////
	void Node_Network::print()
	{
		printf("%s\t ", _name.c_str());
		_operation->print();
		printf("\n\t Inputs: [");
		if (_input_names.size() == 0) {
			printf("]\n");
			return;
		}
		for (int z = 0; z < (int)_input_names.size(); z++) {
			if (z != (int)_input_names.size() - 1) printf(" %s,", _input_names[z].c_str());
			else printf(" %s ]\n", _input_names[z].c_str());
		}
	}
	
} /* namespace lg */
