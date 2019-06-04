// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Variable.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Variable::make(int width)
	{
		return std::shared_ptr<Operation>(new Variable(width));
	}
	
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Variable::make(int width, int height)
	{
		return std::shared_ptr<Operation>(new Variable(width, height));
	}

	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Variable::make(int width, int height, int depth)
	{
		return std::shared_ptr<Operation>(new Variable(width, height, depth));
	}

	////////////////////////////////////////////////////////////
	Variable::Variable(int width) 
	{
		_size = width;
		_width = width;
		_height = 1;
		_depth = 1;
		_outputs.setshape(_size);
        	_outputs.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	Variable::Variable(int width, int height)
	{
		_size = width * height;
		_width = width;
		_height = height;
		_depth = 1;
		_outputs.setshape(width, height);
        	_outputs.fill(0);
	}

	////////////////////////////////////////////////////////////
	Variable::Variable(int width, int height, int depth)
	{
		_size = width * height * depth;
		_width = width;
		_height = height;
		_depth = depth;
		_outputs.setshape(width, height, depth);
        	_outputs.fill(0);
	}

	////////////////////////////////////////////////////////////
	Variable::Variable(lg::IOData& data) 
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
	}
	
	////////////////////////////////////////////////////////////
	void Variable::save(lg::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}

	////////////////////////////////////////////////////////////
	void Variable::print() 
	{
		printf("Type: Variable, Size: %dx%dx%d", _width, _height, _depth);
	}

	////////////////////////////////////////////////////////////
	const Operation::Type Variable::get_type() const
	{
		return Operation::Variable; 
	}

} /* namespace lg */
