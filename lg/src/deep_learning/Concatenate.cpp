// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Concatenate.hpp"
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
	std::shared_ptr<Operation> Concatenate::make()
	{
		return std::shared_ptr<Operation>(new Concatenate());
	}

	////////////////////////////////////////////////////////////
	Concatenate::Concatenate() {}
	
	////////////////////////////////////////////////////////////
	Concatenate::Concatenate(lg::IOData& data)
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
	void Concatenate::initialize(std::vector<Operation*> &inputs)
	{
		//We have to decide the final shape of the tensor
		//and the number of dimensions used
		#define SAME_WIDTH_AND_HEIGHT 0
		#define SAME_WIDTH  1
		#define DIFFERENT_SHAPES 2
		int concatenation_type = SAME_WIDTH_AND_HEIGHT;

		_width = inputs[0]->_outputs.width();
		for (int c = 0; c < (int)inputs.size(); c++) {
			if (_width != inputs[c]->_outputs.width()) {
				concatenation_type = DIFFERENT_SHAPES;
				break;
			}
		}
		
		//All inputs have same width
		//Check if they have same height
		if (concatenation_type == SAME_WIDTH_AND_HEIGHT) {
			_height = inputs[0]->_outputs.height();
			for (int c = 0; c < (int)inputs.size(); c++) {
				if (_height != inputs[c]->_outputs.height()) {
					concatenation_type = SAME_WIDTH;
					break;
				}
			}
		}

		switch (concatenation_type)
		{
			case SAME_WIDTH_AND_HEIGHT:
				_depth = 0;
				for (int c = 0; c < (int)inputs.size(); c++)
					_depth += inputs[c]->_outputs.depth();
				break;
			
			case SAME_WIDTH:
				_height = 0;
				_depth = 1;
				for (int c = 0; c < (int)inputs.size(); c++)
					_height += inputs[c]->_outputs.height();		
				break;

			case DIFFERENT_SHAPES:
				_width = 0;
				_height = 1;
				_depth = 1;
				for (int c = 0; c < (int)inputs.size(); c++)
					_width += inputs[c]->_outputs.width();		
				break;
		}
		
		//Calculate total output size
		_size = _width * _height * _depth;

		//Initialize vectors
        	_outputs.setshape(_width, _height, _depth);
        	_outputs.fill(0);
        	_errors.setshape(_size);
        	_outputs.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	void Concatenate::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Concatenate::run(std::vector<Operation*> &inputs, const bool training) 
	{
		#ifdef CUDA_BACKEND
		
		//Make sure we have the inputs pointers
		if (_inputs_pointers.size() != inputs.size()) {
			_inputs_pointers.setshape(inputs.size());
			std::vector<float*> host_pointers_inputs(inputs.size());
			 for (int c = 0; c < inputs.size(); c++)
			 	host_pointers_inputs[c] = inputs[c]->_outputs.pointer();
			_inputs_pointers.copyToDevice(&host_pointers_inputs[0], (int)host_pointers_inputs.size());
		}
		
		//Store inputs sizes
		if (_pointers_sizes.size() != inputs.size()) {
			std::vector<int> host_pointers_sizes(inputs.size());
			for (int c = 0; c < (int)host_pointers_sizes.size(); c++) 
				host_pointers_sizes[c] = inputs[c]->_outputs.size();
			_pointers_sizes.setshape(inputs.size());
			_pointers_sizes.copyToDevice(&host_pointers_sizes[0], (int)host_pointers_sizes.size());
		}
		
		cuda::concatenate_forward(_inputs_pointers.pointer(), _outputs.pointer(), _pointers_sizes.pointer(), (int)inputs.size(), _size);

		#else
		
		int offset  = 0;
		for (int z = 0; z < (int)inputs.size(); z++) {
			const Tensor_float& in = inputs[z]->_outputs;
			for (int c = 0; c < (int)in.size(); c++)
				_outputs[offset + c] = in[c];
			offset += in.size();
		}
		
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Concatenate::backprop(std::vector<Operation*> &inputs) 
	{
		#ifdef CUDA_BACKEND
		
		//Make sure we have the out errors pointers
		if (_outerrors_pointers.size() != inputs.size()) {
			_outerrors_pointers.setshape(inputs.size());
			std::vector<float*> host_pointers_outerrors(inputs.size());
			 for (int c = 0; c < inputs.size(); c++)
				host_pointers_outerrors[c] = inputs[c]->_errors.pointer();
			_outerrors_pointers.copyToDevice(&host_pointers_outerrors[0], (int)host_pointers_outerrors.size());
		}
	
		cuda::concatenate_backward(_errors.pointer(), _outerrors_pointers.pointer(), _pointers_sizes.pointer(), (int)inputs.size(), _size);

		#else

		int offset  = 0;
		for (int z = 0; z < (int)inputs.size(); z++) {
			Tensor_float& oe = inputs[z]->_errors;
			for (int c = 0; c < (int)oe.size(); c++)
				oe[c] = _errors[offset + c];
			offset += oe.size();
		}

		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Concatenate::get_type() const
	{
		return Operation::Concatenate;
	}
	
	////////////////////////////////////////////////////////////
	void Concatenate::print()
	{
		printf("Type: Concatenate, Output: (%dx%dx%d)", _width, _height, _depth);
	}
	
} /* namespace lg */
