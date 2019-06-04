// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Dropout.hpp"
#include "../util/Util.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#include <stdlib.h>
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Dropout::make(const double drop_probability)
	{
		return std::shared_ptr<Operation>(new Dropout(drop_probability));
	}

	////////////////////////////////////////////////////////////
	Dropout::Dropout(const double drop_probability) 
	{
		_drop_probability = drop_probability;	
	}
	
	////////////////////////////////////////////////////////////
	Dropout::Dropout(lg::IOData& data)
	{
		lg::IOData* size = data.findNode("size");
		ensure(size != NULL);
		lg::IOData* width = data.findNode("width");
		ensure(width != NULL);
		lg::IOData* height = data.findNode("height");
		ensure(height != NULL);
		lg::IOData* depth = data.findNode("depth");
		ensure(depth != NULL);
		lg::IOData* drop_probty = data.findNode("drop_probty");
		ensure(drop_probty != NULL);
		size->get(_size);
		width->get(_width);
		height->get(_height);
		depth->get(_depth);
		drop_probty->get(_drop_probability);
        _outputs.setshape(_width, _height, _depth);
		_outputs.fill(0);
		_errors.setshape(_size);
		_outputs.fill(0);
		
		#ifdef CUDA_BACKEND
		_state_buffer.setshape(_cuda_dropout.getStatesSize() / sizeof(float) + 1);
		_reserve_space_buffer.setshape(_cuda_dropout.getReserveSpaceSize(_size / sizeof(float) + 1));
		_state_buffer.fill(0);
		_reserve_space_buffer.fill(0);
		_cuda_dropout.create(_size, _drop_probability, _state_buffer.pointer()); 
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Dropout::initialize(std::vector<Operation*> &inputs)
	{
		//Calculate size
		_size = inputs[0]->_size;
		_width = inputs[0]->_outputs.width();
		_height = inputs[0]->_outputs.height();
		_depth = inputs[0]->_outputs.depth();

		//Initialize vectors
        _outputs.setshape(_width, _height, _depth);
        _outputs.fill(0);
        _errors.setshape(_size);
        _errors.fill(0);
		
		#ifdef CUDA_BACKEND
		_state_buffer.setshape(_cuda_dropout.getStatesSize() / sizeof(float) + 1);
		_reserve_space_buffer.setshape(_cuda_dropout.getReserveSpaceSize(_size / sizeof(float) + 1));
		_state_buffer.fill(0);
		_reserve_space_buffer.fill(0);
		_cuda_dropout.create(_size, _drop_probability, _state_buffer.pointer()); 
		#endif
    }
	
	////////////////////////////////////////////////////////////
	void Dropout::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
		data.pushNode("drop_probty", _drop_probability);
	}
		
	////////////////////////////////////////////////////////////
	void Dropout::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for errors
		ensure(inputs.size() == 1);

		#ifdef CUDA_BACKEND
	
		_cuda_dropout.forward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _reserve_space_buffer.pointer());
		//cuda::dropout_forward(inputs[0]->_outputs.pointer(), _outputs.pointer(), rand(),
		//	_drop_probability, training, _size);

		//==== TESTING DROPOUT ====
		/*
		lg::Tensor_float t_in(inputs[0]->_outputs.size());
		lg::Tensor_float t_out(_outputs.size());
		inputs[0]->_outputs.copyToHost(&t_in[0], t_in.size());
		_outputs.copyToHost(&t_out[0], t_out.size());
		int in_zero_count = 0;
		int out_zero_count = 0;
		for (int c = 0; c < t_in.size(); c++) if (t_in[c] == 0.f) in_zero_count++;
		for (int c = 0; c < t_out.size(); c++) if (t_out[c] == 0.f) out_zero_count++;
		//printf("%s\n", t_out.tostring().c_str());
		printf("Zeros %d -> %d\n", in_zero_count, out_zero_count);
		*/
		#else

		//Feedforward
		if (training == true)
		{
			for (int z = 0; z < (int)inputs[0]->_outputs.size(); z++)
				_outputs[z] = (lg::util::randf() < _drop_probability) ? 0 : inputs[0]->_outputs[z];
		}
		else
		{
			for (int z = 0; z < (int)inputs[0]->_outputs.size(); z++)
				_outputs[z] = inputs[0]->_outputs[z];
		}
		
		//==== TESTING DROPOUT ====
		/*
		int in_zero_count = 0;
		int out_zero_count = 0;
		for (int c = 0; c < inputs[0]->_outputs.size(); c++) if (inputs[0]->_outputs[c] == 0.f) in_zero_count++;
		for (int c = 0; c < _outputs.size(); c++) if (_outputs[c] == 0.f) out_zero_count++;
		//printf("%s\n", t_out.tostring().c_str());
		printf("Zeros %d -> %d\n", in_zero_count, out_zero_count);
		*/
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Dropout::backprop(std::vector<Operation*> &inputs) 
	{
		//Check errors
		ensure(inputs[0]->_errors.size() > 0);
		
		#ifdef CUDA_BACKEND
	
		_cuda_dropout.backward(_errors.pointer(), inputs[0]->_errors.pointer(), _reserve_space_buffer.pointer());
		//cuda::dropout_backward(_errors.pointer(), inputs[0]->_errors.pointer(), _outputs.pointer(), _drop_probability, _size);

		#else
		
		//Shortcuts
		Tensor_float &out_errors = inputs[0]->_errors;

		//Feedforward
		for (int z = 0; z < (int)out_errors.size(); z++)
			out_errors[z] = (_outputs[z] == 0) ? 0 :  _errors[z] * (1 - _drop_probability);

		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Dropout::get_type() const
	{
		return Operation::Dropout;
	}
	
	////////////////////////////////////////////////////////////
	void Dropout::print()
	{
		printf("Type: Dropout, Size: %d, Drop_probability: %f", _size, _drop_probability);
	}
	
} /* namespace lg */
