// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Selu.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

static const double _alpha = 1.6732632423543772;
static const double _scale = 1.0507009873554804;

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Selu::make()
	{
		return std::shared_ptr<Operation>(new Selu());
	}

	////////////////////////////////////////////////////////////
	Selu::Selu() {}

	////////////////////////////////////////////////////////////
	Selu::Selu(lg::IOData& data)
	{
		lg::IOData* size = data.findNode("size");
		ensure(size != NULL);
		lg::IOData* width = data.findNode("width");
		ensure(width != NULL);
		lg::IOData* height = data.findNode("height");
		ensure(height != NULL);
		lg::IOData* depth = data.findNode("depth");
		ensure(depth != NULL);
		size->get(_size);
		width->get(_width);
		height->get(_height);
		depth->get(_depth);
		_outputs.setshape(_width, _height, _depth);
		_outputs.fill(0);
		_errors.setshape(_size);
		_outputs.fill(0);

#ifdef CUDA_BACKEND
		_cudnnactivation.create(_size, 1, lg::cudnn::ACTIVATION_RELU);
#endif
	}

	////////////////////////////////////////////////////////////
	void Selu::initialize(std::vector<Operation*> &inputs)
	{
		//Check for errors
		ensure(inputs.size() == 1);

		//Calculate size
		_size = inputs[0]->_outputs.size();
		_width = inputs[0]->_outputs.width();
		_height = inputs[0]->_outputs.height();
		_depth = inputs[0]->_outputs.depth();

		//Initialize vectors
		_outputs.setshape(_width, _height, _depth);
		_outputs.fill(0);
		_errors.setshape(_size);
		_outputs.fill(0);

#ifdef CUDA_BACKEND
		_cudnnactivation.create(_size, 1, lg::cudnn::ACTIVATION_RELU);
#endif
	}

	////////////////////////////////////////////////////////////
	void Selu::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}

	////////////////////////////////////////////////////////////
	void Selu::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		ensure(inputs[0]->_outputs.size() == _outputs.size());

#ifdef CUDA_BACKEND

		cuda::selu_forward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _size);

#else
		//Shortcuts
		const Tensor_float& in = inputs[0]->_outputs;

		//Feedforward
		for (int c = 0; c < (int)_outputs.size(); c++) {
			if (in[c] >= 0.0) _outputs[c] = _scale * in[c];
			else _outputs[c] = _scale * (_alpha * exp(in[c]) - _alpha);
		}
#endif
	}

	////////////////////////////////////////////////////////////
	void Selu::backprop(std::vector<Operation*> &inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);

#ifdef CUDA_BACKEND

		cuda::selu_backward(_errors.pointer(), inputs[0]->_errors.pointer(), _outputs.pointer(), _size);

#else

		//Shortcuts
		Tensor_float &out_errors = inputs[0]->_errors;

		//Feedforward
		for (int c = 0; c < (int)out_errors.size(); c++) {
			if (_outputs[c] >= 0.0) out_errors[c] = _scale * _errors[c];
			else out_errors[c] = _errors[c] * (_outputs[c] + _scale * _alpha);
		}
#endif
	}

	////////////////////////////////////////////////////////////
	const Operation::Type Selu::get_type() const
	{
		return Operation::Selu;
	}

	////////////////////////////////////////////////////////////
	void Selu::print()
	{
		printf("Type: Selu, Size: %d", _size);
	}

#ifndef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Selu::forward(const Tensor_float input, Tensor_float output)
	{
		//Check for errors
		ensure(output.size() == input.size());

		//Feedforward
		for (int c = 0; c < (int)output.size(); c++) {
			if (input[c] >= 0.0) output[c] = _scale * input[c];
			else output[c] = _scale * (_alpha * exp(input[c]) - _alpha);
		}
	}

	////////////////////////////////////////////////////////////
	void Selu::backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors)
	{
		//Check for errors
		ensure(errors.size() == out_errors.size());

		//Backward
		for (int c = 0; c < (int)out_errors.size(); c++) {
			if (outputs[c] > 0.0) out_errors[c] = _scale * errors[c];
			else out_errors[c] = errors[c] * (outputs[c] + _scale * _alpha);
		}
	}

#endif

} /* namespace lg */
