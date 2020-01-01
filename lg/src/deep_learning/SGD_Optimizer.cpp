// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "SGD_Optimizer.hpp"
#include "Neural_Network.hpp"
#include "../util/ensure.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	
	////////////////////////////////////////////////////////////
	SGD_Optimizer::SGD_Optimizer()
	{
		_current_sample = 0;
		_learningrate = 0.1;
		_momentum = 0;
		_batch_size = 1;
		_costfunction = Cost(Cost::SquaredError);
	}
	
	////////////////////////////////////////////////////////////
	SGD_Optimizer::SGD_Optimizer(const int batch_size, const double learningrate, const double momentum,
					const Cost::CostType cost_function)
	{
		_current_sample = 0;
		_learningrate = learningrate;
		_momentum = momentum;
		_batch_size = batch_size;
		_costfunction = Cost(cost_function);
	}
	
	#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void SGD_Optimizer::fit(Neural_Network& net, CUDA_Tensor_float &inputs, CUDA_Tensor_float &targets)
	{
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
	}
	
	#else

	////////////////////////////////////////////////////////////
	void SGD_Optimizer::fit(Neural_Network& net, Tensor_float &inputs, Tensor_float &targets)
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

		//Accumulate deltas
		for (int c = (int)nodes.size()-1; c >= 0; c--) {
			
			//Backpropagate errors
			nodes[c].backprop();

			//Calculate deltas
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
		
		/*
		bool nan_f = false;
		for (int c = 0; c < nodes.size(); c++) {
			if (nodes[c].getOperation()->_outputs.isNaN()) {
				printf("Node %d Output NaN\n", c);
				nan_f = true;
			}
			float maxval;
			nodes[c].getOperation()->_outputs.max(&maxval, NULL);
			if (maxval > 250.f) {
				printf("Node %d Max %f\n", c, maxval);
				nan_f = true;
			}
		}
		for (int c = (int)nodes.size()-1; c >= 0; c--) {
			if (nodes[c].getOperation()->_errors.isNaN()) {
				printf("Node %d errors NaN\n", c);
				nan_f = true;
			}
			float maxval;
			nodes[c].getOperation()->_errors.max(&maxval, NULL);
			float minval;
			nodes[c].getOperation()->_errors.min(&minval, NULL);
			maxval = maxval > -minval ? maxval : -minval;
			if (maxval > 10.f) {
				printf("Node %d MaxErr %f\n", c, maxval);
				printf("%s\n", nodes.back().getOperation()->_outputs.tostring().c_str());
				printf("%s\n", targets.tostring().c_str());
				printf("%s\n", nodes.back().getOperation()->_errors.tostring().c_str());
				nan_f = true;
			}
		}
		if (nan_f == true) {
			for (int c = 0; c < nodes.front().getOperation()->_outputs.size(); c++) {
				printf("%f\n", nodes.front().getOperation()->_outputs[c]);	
			}

			//printf("%s\n", nodes.front().getOperation()->_outputs.tostring().c_str())
		}
		ensure(nan_f == false);
		*/
	}

	#endif

} /* namespace lg */
