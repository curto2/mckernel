// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "CapsulesDense.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#include "WeightRegularization.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif
#include "Cost.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg 
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> CapsulesDense::make(const int size, bool use_bias,
			const float gradient_clipping, float l1_regularization, float l2_regularization)
	{
		return std::shared_ptr<Operation>(new CapsulesDense(size, use_bias, gradient_clipping, l1_regularization, l2_regularization));
	}

	////////////////////////////////////////////////////////////
	CapsulesDense::CapsulesDense()
	{
		_size = 0;
		_use_bias = true;
		_l1_regularization = 0;
		_l2_regularization = 0;
		_fixed_parameters = false;
	}

	////////////////////////////////////////////////////////////
	CapsulesDense::CapsulesDense(const int size, bool use_bias, const float gradient_clipping, float l1_regularization, float l2_regularization)
	{
		_size = size;
		_use_bias = use_bias;
		ensure(l1_regularization >= 0 && l1_regularization < 1);
		ensure(l2_regularization >= 0 && l2_regularization < 1);
		_gradient_clipping = gradient_clipping;
		_l1_regularization = l1_regularization;
		_l2_regularization = l2_regularization;
		_fixed_parameters = false;
	}

	////////////////////////////////////////////////////////////
	CapsulesDense::CapsulesDense(lg::IOData& data)
	{
		lg::IOData* size = data.findNode("size");
		ensure(size != NULL);
		lg::IOData* input_size = data.findNode("input_size");
		ensure(input_size != NULL);
		lg::IOData* use_bias = data.findNode("use_bias");
		ensure(use_bias != NULL);
		size->get(_size);
		input_size->get(_input_size);
		use_bias->get(_use_bias);

		lg::IOData* l1 = data.findNode("l1_regularization");
		lg::IOData* l2 = data.findNode("l2_regularization");
		if (l1 != NULL) l1->get(_l1_regularization);
		else _l1_regularization = 0; //default value
		if (l2 != NULL) l1->get(_l2_regularization);
		else _l2_regularization = 0; //default value

		lg::IOData* g_clipping = data.findNode("gradient_clipping");
		if (g_clipping != NULL) g_clipping->get(_gradient_clipping);

		_weights.load(data, "weights");
		_bias.load(data, "bias");

		_outputs.setshape(_size);
		_outputs.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
		_deltas.setshape(_size * (_input_size + 1));
		_deltas.fill(0);
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("input_size", _input_size);
		data.pushNode("use_bias", _use_bias);
		data.pushNode("l1_regularization", _l1_regularization);
		data.pushNode("l2_regularization", _l2_regularization);
		data.pushNode("gradient_clipping", _gradient_clipping);
		_weights.save(data, "weights");
		_bias.save(data, "bias");
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::initialize(std::vector<Operation*> &inputs)
	{
		//Calculate input size
		int input_size = 0;
		for (int c = 0; c < (int)inputs.size(); c++)
			input_size += inputs[c]->_outputs.size();

		initialize(input_size);
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::initialize(int input_size)
	{
		_input_size = input_size;

		//Initialize variables and buffers
		_outputs.setshape(_size);
		_outputs.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
		_deltas.setshape(_size * (_input_size + 1));
		_deltas.fill(0);

		//Initialize weights
		printf("%f\n", sqrt(1.f / (_input_size + _size)));
		_weights.setshape(_size, _input_size);
		_weights.fill(0.0, sqrt(0.1f / (_input_size + _size)));
		_bias.setshape(_size);
		_bias.fill(0.0, sqrt(0.1f / (_input_size + _size)));
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::run(std::vector<Operation*> &inputs, const bool training) 
	{
		run(inputs[0]->_outputs, false);
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::backprop(std::vector<Operation*> &inputs) 
	{
		backprop(inputs[0]->_errors);	
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		accumulate_deltas(inputs[0]->_outputs);
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::update_parameters(const float learningrate)
	{
		if (_fixed_parameters == true) return;

#ifdef CUDA_BACKEND
		
		//Gradient clipping
		if (_gradient_clipping != 0)
			cuda::gradient_clipping(_deltas.pointer(), _deltas.size(), _gradient_clipping);

		//Update weights using gradients
		cuda::linear_update_parameters(_weights.pointer(), _bias.pointer(), _deltas.pointer(),
				learningrate, _input_size, _size);
		
		//Regularization penalty
		if (_l1_regularization != 0) lg::wrn::l1_regularization(_weights, _l1_regularization, learningrate);
		else if (_l2_regularization != 0) lg::wrn::l2_regularization(_weights, _l2_regularization, learningrate);

#else
		
		//Gradient clipping
		if (_gradient_clipping != 0) {
			for (int c = 0; c < (int)_deltas.size(); c++) {
				if (_deltas[c] > _gradient_clipping) _deltas[c] = _gradient_clipping;
				else if(_deltas[c] < -_gradient_clipping) _deltas[c] = -_gradient_clipping;
			}
		}

		//Update weights using gradients
		int d = 0;
		for (int c = 0; c < _weights.width(); c++) {
			for (int k = 0; k <	_weights.height(); k++)
				_weights.at(k, c) += _deltas[d++] * learningrate;
			_bias[c] += _deltas[d++] * learningrate;
		}

		//Regularization penalty
		if (_l1_regularization != 0) lg::wrn::l1_regularization(_weights, _l1_regularization, learningrate);
		else if (_l2_regularization != 0) lg::wrn::l2_regularization(_weights, _l2_regularization, learningrate);

#endif
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::reset_deltas(const double momentum)
	{
#ifdef CUDA_BACKEND

		if (_deltas.size() > 0) CUDA_Tensor_float_scale(_deltas, momentum);

#else

		for (int c = 0; c < _deltas.size(); c++)
			_deltas[c] *= momentum;

#endif
	}

#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void CapsulesDense::run(const CUDA_Tensor_float& input, bool accumulate)
	{
		cuda::linear_forward(_weights.pointer(), _bias.pointer(), input.pointer(),
				_outputs.pointer(), input.size(), _outputs.size(), accumulate, _use_bias);
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::backprop(CUDA_Tensor_float& out_errors)
	{
		cuda::linear_backward(_weights.pointer(), out_errors.pointer(), _errors.pointer(),
				out_errors.size(), _errors.size());
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::accumulate_deltas(const CUDA_Tensor_float& input)
	{
		cuda::linear_accumulate_deltas(_deltas.pointer(), input.pointer(), _errors.pointer(),
				input.size(), _errors.size(), _use_bias);
	}

#else

	////////////////////////////////////////////////////////////
	void CapsulesDense::run(const Tensor_float input, bool accumulate)
	{
		if (_use_bias)
		{
			if (!accumulate)
			{
				//Reset outputs with bias
				for (int c = 0; c < _outputs.size(); c++)
					_outputs[c] = _bias[c];
			}
			else
			{
				//Add bias to output
				for (int c = 0; c < _outputs.size(); c++)
					_outputs[c] += _bias[c];
			}
		}
		else
		{
			if (!accumulate)
			{
				for (int c = 0; c < _outputs.size(); c++)
					_outputs[c] = 0;
			}
		}

		//Compute all inputs
		int weight_index = 0;
		for (int c = 0; c < input.size(); c++) {
			if (input[c] == 0) continue;
			weight_index = c * _outputs.size();
			for (int k = 0; k < _outputs.size(); k++)
				_outputs[k] += _weights[weight_index++] * input[c];
		}
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::backprop(Tensor_float out_errors)
	{
		//Check we must have only one input
		if (out_errors.size() == 0) return;

		//Back-propagate errors
		for (int c = 0; c < _errors.size(); c++) {
			if (_errors[c] == 0) continue;
			for (int k = 0; k < _weights.height(); k++)
				out_errors[k] += _weights.at(k, c) * _errors[c];
		}
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::setFixedParameters(const bool fixedparameters)
	{
		_fixed_parameters = fixedparameters;
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::accumulate_deltas(const Tensor_float input)
	{
		int d = 0;
		for (int c = 0; c < _errors.size(); c++) {
			for (int k = 0; k <	input.size(); k++)
				_deltas[d++] += input[k] * _errors[c];
			if (_use_bias) _deltas[d++] += _errors[c];
			else d++;
		}
	}

#endif

	////////////////////////////////////////////////////////////
	void CapsulesDense::reset_outputs()
	{
		_outputs.fill(0);	
	}

#ifndef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void CapsulesDense::gradient_check()
	{
	}

#endif

	////////////////////////////////////////////////////////////
	const Operation::Type CapsulesDense::get_type() const
	{
		return Operation::CapsulesDense;
	}

	////////////////////////////////////////////////////////////
	void CapsulesDense::print()
	{
		printf("Type: CapsulesDense, Size: %d, Input_Size: %d, Weights: %d, Bias: %d, l1: %f, l2: %f",
				_size, _input_size, _size * (_input_size + 1), (int)_use_bias, _l1_regularization, _l2_regularization);
	}

} /* namespace lg */
