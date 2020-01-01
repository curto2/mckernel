// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Relu.hpp"
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
	std::shared_ptr<Operation> Relu::make()
	{
		return std::shared_ptr<Operation>(new Relu());
	}

	////////////////////////////////////////////////////////////
	Relu::Relu() {}
	
	////////////////////////////////////////////////////////////
	Relu::Relu(lg::IOData& data)
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
	void Relu::initialize(std::vector<Operation*> &inputs)
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
	void Relu::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Relu::run(std::vector<Operation*> &inputs, const bool training) 
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
			_outputs[c] = in[c] * (in[c] > 0);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Relu::backprop(std::vector<Operation*> &inputs) 
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
			out_errors[c] = _errors[c] * (_outputs[c] > 0);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Relu::get_type() const
	{
		return Operation::Relu;
	}
	
	////////////////////////////////////////////////////////////
	void Relu::print()
	{
		printf("Type: Relu, Size: %d", _size);
	}
	
	#ifndef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	void Relu::forward(const Tensor_float input, Tensor_float output)
	{
		//Check for errors
		ensure(output.size() == input.size());

		//Feedforward
		for (int c = 0; c < (int)output.size(); c++)
			output[c] = input[c] * (input[c] > 0);
	}

	////////////////////////////////////////////////////////////
	void Relu::backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors)
	{
		//Check for errors
		ensure(errors.size() == out_errors.size());

		//Backward
		for (int c = 0; c < (int)out_errors.size(); c++)
			out_errors[c] = errors[c] * (outputs[c] > 0);
	}
	
	#endif
	
} /* namespace lg */
