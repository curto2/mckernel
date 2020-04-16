// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Tanh.hpp"
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
	std::shared_ptr<Operation> Tanh::make()
	{
		return std::shared_ptr<Operation>(new Tanh());
	}

	////////////////////////////////////////////////////////////
	Tanh::Tanh() {} 
	
	////////////////////////////////////////////////////////////
	Tanh::Tanh(lg::IOData& data)
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
		_cudnnactivation.create(_size, 1, lg::cudnn::ACTIVATION_TANH);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::initialize(std::vector<Operation*> &inputs)
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
		_cudnnactivation.create(_size, 1, lg::cudnn::ACTIVATION_TANH);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Tanh::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		
		//Check for errors
		ensure(_outputs.size() == inputs[0]->_outputs.size());

		#ifdef CUDA_BACKEND
	
		_cudnnactivation.forward(inputs[0]->_outputs.pointer(), _outputs.pointer());

		#else

		forward(inputs[0]->_outputs, _outputs);

		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::backprop(std::vector<Operation*> &inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		
		//Check for errors
		ensure(_errors.size() == inputs[0]->_errors.size());

		#ifdef CUDA_BACKEND
	
		_cudnnactivation.backward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _errors.pointer(), inputs[0]->_errors.pointer());

		#else
		
		backward(_errors, _outputs, inputs[0]->_errors);

		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Tanh::get_type() const
	{
		return Operation::Tanh; 
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::print()
	{
		printf("Type: Tanh, Size: %d", _size);
	}
	
	////////////////////////////////////////////////////////////
	///	RAW OPERATIONS
	////////////////////////////////////////////////////////////
	
	#ifndef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Tanh::forward(const Tensor_float input, Tensor_float output)
	{
		//Feedforward
		for (int z = 0; z < (int)output.size(); z++)
			output[z] = tanh(input[z]);
	}

	////////////////////////////////////////////////////////////
	void Tanh::backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors)
	{
		//Backward
		for (int z = 0; z < (int)out_errors.size(); z++)
			out_errors[z] = (1.f - outputs[z] * outputs[z]) * errors[z];
	}
	
	#endif
	
} /* namespace lg */
