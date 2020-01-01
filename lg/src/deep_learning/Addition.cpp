// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Addition.hpp"
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
	std::shared_ptr<Operation> Addition::make()
	{
		return std::shared_ptr<Operation>(new Addition());
	}
	
	////////////////////////////////////////////////////////////
	Addition::Addition() {}
	
	////////////////////////////////////////////////////////////
	Addition::Addition(lg::IOData& data)
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
	}
	
	////////////////////////////////////////////////////////////
	void Addition::initialize(std::vector<Operation*> &inputs)
	{
		//Check for errors
		ensure(inputs.size() > 0);
		_size = inputs[0]->_outputs.size();
		for (int c = 0; c < (int)inputs.size(); c++)
			ensure_print(_size == inputs[c]->_outputs.size(), "%d %d\n", _size, inputs[c]->_outputs.size());
		
		//Calculate size
		_width = inputs[0]->_outputs.width();
		_height = inputs[0]->_outputs.height();
		_depth = inputs[0]->_outputs.depth();

		//Initialize vectors
        	_outputs.setshape(_width, _height, _depth);
		_outputs.fill(0);
		_errors.setshape(_size);
		_outputs.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	void Addition::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Addition::run(std::vector<Operation*> &inputs, const bool training) 
	{
		#ifdef CUDA_BACKEND
	
		for (int c = 0; c < (int)inputs.size(); c++) {
			if (c == 0) CUDA_Tensor_float_copy(inputs[c]->_outputs, _outputs);
			else CUDA_Tensor_float_sum(inputs[c]->_outputs, _outputs);
		}

		#else
		for (int c = 0; c < (int)inputs.size(); c++) {
			
			//Shortcut
			const Tensor_float& in = inputs[c]->_outputs;
			
			if (c == 0)
			{
				for (int z = 0; z < _size; z++)
					_outputs[z] = in[z];
			}
			else
			{
				for (int z = 0; z < _size; z++)
					_outputs[z] += in[z];
			}
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Addition::backprop(std::vector<Operation*> &inputs) 
	{
		#ifdef CUDA_BACKEND

		for (int c = 0; c < (int)inputs.size(); c++)
			CUDA_Tensor_float_sum(_errors, inputs[c]->_errors);

		#else
		for (int c = 0; c < (int)inputs.size(); c++) {
			
			//Shortcuts
			Tensor_float &out_errors = inputs[c]->_errors;
			
			//Feedforward
			for (int c = 0; c < _size; c++)
				out_errors[c] += _errors[c];
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Addition::get_type() const
	{
		return Operation::Addition;
	}
	
	////////////////////////////////////////////////////////////
	void Addition::print()
	{
		printf("Type: Addition, Size: %d", _size);
	}
	
} /* namespace lg */
