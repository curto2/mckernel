// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "DFA_Optimizer.hpp"
#include "Neural_Network.hpp"
#include "../util/ensure.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	
	////////////////////////////////////////////////////////////
	DFA_Optimizer::DFA_Optimizer()
	{
		_current_sample = 0;
		_learningrate = 0.1;
		_momentum = 0;
		_batch_size = 1;
		_costfunction = Cost::SquaredError;
	}
	
	////////////////////////////////////////////////////////////
	DFA_Optimizer::DFA_Optimizer(const int batch_size, const double learningrate, const double momentum,
					const Cost::CostType cost_function)
	{
		_current_sample = 0;
		_learningrate = learningrate;
		_momentum = momentum;
		_batch_size = batch_size;
		_costfunction = cost_function;
		_feedback_weights.setshape(200 * 10);
		_feedback_weights.fill(0, 0.5);
		for (int c = 0; c < _feedback_weights.size(); c++) {
			//if (rand() % 1000 < 700) _feedback_weights[c] = 0;
			//if (_feedback_weights[c] < 0) _feedback_weights[c] = -1;
			//else _feedback_weights[c] = 1; 
		}
		_feedback_errors.setshape(200);
		_feedback_errors.fill(0);
	}
	
	#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void DFA_Optimizer::fit(Neural_Network& net, CUDA_Tensor_float &inputs, CUDA_Tensor_float &targets)
	{
		/*
		ensure(targets.depth() == 1 && targets.height() == 1);
		
		//Feedforward
		net.run(inputs, true);
		
		//Shortcut
		std::vector< Node_Network > &nodes = net.getNodes();

		//Reset errors
		for (int c = 0; c < (int)nodes.size(); c++)
			nodes[c].reset_errors();
		
		//Calculate cost on GPU
		_costfunction.getDeltacuda(nodes.back().getOperation()->_outputs, targets, nodes.back().getOperation()->_errors);
		_error = _costfunction.getErrorcuda(nodes.back().getOperation()->_outputs, targets);

		//Accumulate deltas
		for (int c = (int)nodes.size()-1; c >= 0; c--) {
			nodes[c].backprop();
			nodes[c].accumulate_deltas();
		}
		
		//Update weights if we reach the batch size
		if (++_current_sample >= _batch_size) {
			
			//Update weights and reset deltas
			for (int c = 0; c < (int)nodes.size(); c++) {
				nodes[c].update_parameters(_learningrate);
				nodes[c].reset_deltas(_momentum);
			}

			_current_sample = 0;
		}
		*/
	}
	
	#else

	////////////////////////////////////////////////////////////
	void DFA_Optimizer::fit(Neural_Network& net, Tensor_float &inputs, Tensor_float &targets)
	{
		ensure(targets.depth() == 1 && targets.height() == 1);
		
		//Feedforward
		net.run(inputs, true);
			
		//Shortcuts
		std::vector< Node_Network >& nodes = net.getNodes();

		//Reset errors
		for (int c = 0; c < (int)nodes.size(); c++)
			nodes[c].reset_errors();
		
		//Calculate cost on host
		_costfunction.getDelta(nodes.back().getOperation()->_outputs, targets, nodes.back().getOperation()->_errors);
		_error = _costfunction.getError(nodes.back().getOperation()->_outputs, targets);
		
		const Tensor_float& errs = nodes.back().getOperation()->_errors;
				
		const float scale = 0.09;
		int l = 0;
		for (int k = 0; k < _feedback_errors.size(); k++) {
			_feedback_errors[k] = 0;
			for (int z = 0; z < errs.size(); z++) {
				_feedback_errors[k] += errs[z] * _feedback_weights[l++] * scale;
			}
		}
		
		nodes.back().backprop();

		//Accumulate deltas
		l = 0;
		for (int c = (int)nodes.size()-1; c >= 0 ; c--) {
			Operation* op = nodes[c].getOperation();
			if (op->get_type() == Operation::Softmax ||
				op->get_type() == Operation::Sigmoid ||
				op->get_type() == Operation::Relu ||
				op->get_type() == Operation::Tanh ||
				op->get_type() == Operation::Averagepooling ||
				op->get_type() == Operation::Maxpooling)
			{

				Tensor_float& node_errors = nodes[c].getOperation()->_errors;
				for (int k = 0; k < node_errors.size(); k++) {
					node_errors[k] += _feedback_errors[l++];
					if (l >= _feedback_errors.size()) l = 0;
				}

				nodes[c].backprop();
			}
			else if (op->get_type() == Operation::Normalization)
			{
				nodes[c].backprop();
			}

			//Calculate deltas
			nodes[c].accumulate_deltas();
		}
		
		/*
		nodes.back().backprop();

		//Accumulate deltas
		int l = 1;
		for (int c = (int)nodes.size()-1; c >= 0 ; c--) {
			Operation* op = nodes[c].getOperation();
			if (op->get_type() == Operation::Softmax ||
				op->get_type() == Operation::Sigmoid ||
				op->get_type() == Operation::Relu ||
				op->get_type() == Operation::Tanh ||
				op->get_type() == Operation::Averagepooling ||
				op->get_type() == Operation::Maxpooling)
			{

				for (int z = 0; z < errs.size(); z++) {
					Tensor_float& node_errors = nodes[c].getOperation()->_errors;
					float scale = 0.01;
					for (int k = 0; k < node_errors.size(); k++) {
						node_errors[k] += errs[z] * _feedback_weights[l++] * scale;
						if (l >= _feedback_weights.size()) l = 0;
					}
				}

				nodes[c].backprop();
			}
			else if (op->get_type() == Operation::Normalization)
			{
				nodes[c].backprop();
			}

			//Calculate deltas
			nodes[c].accumulate_deltas();
		}
		*/
		
		//Update weights if we reach the batch size
		if (++_current_sample >= _batch_size) {
			
			//Update weights and reset deltas
			for (int c = 0; c < (int)nodes.size(); c++) {
				nodes[c].update_parameters(_learningrate);
				nodes[c].reset_deltas(_momentum);
			}

			_current_sample = 0;
		}
	}

	#endif

} /* namespace lg */
