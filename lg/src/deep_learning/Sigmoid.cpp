// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Sigmoid.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Sigmoid::make()
	{
		return std::shared_ptr<Operation>(new Sigmoid());
	}

	////////////////////////////////////////////////////////////
	Sigmoid::Sigmoid() {}
	
	////////////////////////////////////////////////////////////
	Sigmoid::Sigmoid(lg::IOData& data)
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
		_errors.fill(0);
		
		#ifdef CUDA_BACKEND
		_cudnnactivation.create(_size, 1, lg::cudnn::ACTIVATION_SIGMOID);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Sigmoid::initialize(std::vector<Operation*> &inputs)
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
        	_errors.fill(0);
		
		#ifdef CUDA_BACKEND
		_cudnnactivation.create(_size, 1, lg::cudnn::ACTIVATION_SIGMOID);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Sigmoid::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Sigmoid::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		ensure(inputs[0]->_outputs.size() == _outputs.size());
		
		#ifdef CUDA_BACKEND
		
		_cudnnactivation.forward(inputs[0]->_outputs.pointer(), _outputs.pointer());

		#else
		//Shortcuts
		const Tensor_float& in = inputs[0]->_outputs;

		//Feedforward
		for (int c = 0; c < (int)_outputs.size(); c++)
			_outputs[c] = 1.0 / (1.0 + exp(-in[c]));
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Sigmoid::backprop(std::vector<Operation*> &inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		
		#ifdef CUDA_BACKEND
		
		_cudnnactivation.backward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _errors.pointer(), inputs[0]->_errors.pointer());

		#else
		//Shortcuts
		Tensor_float &out_errors = inputs[0]->_errors;
		
		//Feedforward
		for (int c = 0; c < (int)out_errors.size(); c++)
			out_errors[c] = _outputs[c] * (1.f - _outputs[c]) * _errors[c];
		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Sigmoid::get_type() const
	{
		return Operation::Sigmoid;
	}
	
	////////////////////////////////////////////////////////////
	void Sigmoid::print()
	{
		printf("Type: Sigmoid, Size: %d", _size);
	}
	
	#ifndef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	void Sigmoid::forward(const Tensor_float input, Tensor_float output)
	{
		//Check for errors
		ensure(output.size() == input.size());

		//Feedforward
		for (int c = 0; c < (int)output.size(); c++)
			output[c] = 1.0 / (1.0 + exp(-input[c]));
	}

	////////////////////////////////////////////////////////////
	void Sigmoid::backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors)
	{
		//Check for errors
		ensure(errors.size() == out_errors.size());

		//Backward
		for (int c = 0; c < (int)out_errors.size(); c++)
			out_errors[c] = outputs[c] * (1.f - outputs[c]) * errors[c];
	}

	#endif
	
} /* namespace lg */
