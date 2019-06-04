// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Partial.hpp"
#include "../util/Util.hpp"
#include <math.h>
#include "../util/ensure.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg 
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Partial::make(const int size, const double connectivity)
	{
		return std::shared_ptr<Operation>(new Partial(size, connectivity)); 
	}
	
	////////////////////////////////////////////////////////////
	Partial::Partial(const int size, const double connectivity)
	{
		_size = size;
		_connectivity = connectivity;
	}
	
	////////////////////////////////////////////////////////////
	Partial::Partial(lg::IOData& data)
	{
		/*
		TODO
		file.read(reinterpret_cast<char*>(&_size), sizeof(_size));
		file.read(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.read(reinterpret_cast<char*>(&_connectivity), sizeof(_connectivity));
		
		int _forward_map_size;
		file.read(reinterpret_cast<char*>(&_forward_map_size), sizeof(_forward_map_size));
		_forward_map = std::vector< std::vector< int > >(_forward_map_size);
		for (int z = 0; z < _forward_map_size; z++) {
			int _map_size; 
			file.read(reinterpret_cast<char*>(&_map_size), sizeof(_map_size));
			_forward_map[z] = std::vector<int>(_map_size);
			file.read(reinterpret_cast<char*>(&_forward_map[z][0]), sizeof(int) * _map_size);
		}
		
		_outputs.setshape(_size);
        	_outputs.fill(0);
		_errors.setshape(_size);
        	_errors.fill(0);
		_deltas.setshape(_outputs.size() * (_input_size + 1));
        	_deltas.fill(0);
		_weights.load(file);
		_bias.load(file);
		*/
	}
	
	////////////////////////////////////////////////////////////
	void Partial::save(lg::IOData& data)
	{
		/*
		TODO
		file.write(reinterpret_cast<char*>(&_size), sizeof(_size));
		file.write(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.write(reinterpret_cast<char*>(&_connectivity), sizeof(_connectivity));
		int _forward_map_size = _forward_map.size();
		file.write(reinterpret_cast<char*>(&_forward_map_size), sizeof(_forward_map_size));
		for (int z = 0; z < _forward_map_size; z++) {
			int _map_size = _forward_map[z].size(); 
			file.write(reinterpret_cast<char*>(&_map_size), sizeof(_map_size));
			file.write(reinterpret_cast<char*>(&_forward_map[z][0]), sizeof(int) * _map_size);
		}
		_weights.save(file);
		_bias.save(file);
		*/
	}
	
	////////////////////////////////////////////////////////////
	void Partial::initialize(std::vector<Operation*> &inputs)
	{
		//We can have only one input
		ensure(inputs.size() == 1);

		//Calculate input size
		_input_size = 0;
		for (int z = 0; z < (int)inputs.size(); z++)
			_input_size += inputs[z]->_outputs.size();
		
		//Initialize variables and buffers
        	_outputs.setshape(_size);
        	_outputs.fill(0);
        	_errors.setshape(_size);
        	_errors.fill(0);
        	_deltas.setshape(_outputs.size() * (_input_size + 1));
        	_deltas.fill(0);
		_forward_map = std::vector< std::vector< int > >(_input_size);
		/*
		for (int z = 0; z < _input_size; z++) {
			_forward_map[z] = std::vector<int>();
			for (int k = 0; k < _size; k++)
				if (lg::util::randf() < _connectivity)
					_forward_map[z].push_back(k);
		}
		*/
		for (int z = 0; z < _input_size; z++) {
			_forward_map[z] = std::vector<int>();
			for (int k = 0; k < _size; k++)
				_forward_map[z].push_back(k);
		}

		//Initialize weights
        	_weights.setshape(_size, _input_size);
        	_weights.fill(0.0, 6.0 / sqrt(_input_size + _size));
        	_bias.setshape(_size);
		_bias.fill(0.0, 6.0 / sqrt(_input_size + _size));
	}
	
	////////////////////////////////////////////////////////////
	void Partial::run(std::vector<Operation*> &inputs, const bool training) 
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		//Shortcuts
		const Tensor_float &in = inputs[0]->_outputs;

		//Reset outputs with bias
		for (int z = 0; z < _size; z++)
			_outputs[z] = _bias[z];
		
		//Compute all inputs
		for (int z = 0; z < in.size(); z++) {
			if (in[z] == 0) continue;
			for (int k = 0; k < (int)_forward_map[z].size(); k++) {
				_outputs[_forward_map[z][k]] += _weights.at(z, _forward_map[z][k]) * in[z];
			}
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::backprop(std::vector<Operation*> &inputs) 
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		//Check we must have only one input
		Tensor_float &out_errors = inputs[0]->_errors;
		if (out_errors.size() == 0) return;

		//Backpropagate errors
		for (int z = 0; z < out_errors.size(); z++) {
			for (int k = 0; k < (int)_forward_map[z].size(); k++) {
				out_errors[z] += _weights.at(z, _forward_map[z][k]) * _errors[_forward_map[z][k]];
			}
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		const Tensor_float &in = inputs[0]->_outputs;
		
		/*
		int d = 0;
		for (int z = 0; z < _weights.width(); z++) {
			for (int k = 0; k <	_weights.height(); k++)
				_deltas[d++] += in[k] * _errors[z];
			_deltas[d++] += _errors[z];
		}*/
		
		for (int z = 0; z < _weights.height(); z++)
			for (int k = 0; k < _forward_map[z].size(); k++)
				_deltas[z * _size + _forward_map[z][k]] += in[z] * _errors[_forward_map[z][k]];

		for (int k = 0; k < _size; k++)
			_deltas[_size * _input_size + k] += _errors[k];

		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::update_parameters(const float learningrate)
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		/*
		int d = 0;
		for (int z = 0; z < _weights.width(); z++) {
			for (int k = 0; k <	_weights.height(); k++)
				_weights.at(k, z) += _deltas[d++] * learningrate;
			_bias[z] += _deltas[d++] * learningrate;
		}*/

		for (int z = 0; z < _weights.height(); z++)
			for (unsigned int k = 0; k < _forward_map[z].size(); k++)
				_weights.at(z, _forward_map[z][k]) += _deltas[z * _size + _forward_map[z][k]] * learningrate;

		for (int k = 0; k < _size; k++)
			_bias[k] += _deltas[_size * _input_size + k] * learningrate;
		#endif
	}

	////////////////////////////////////////////////////////////
	void Partial::pruning(float alpha)
	{
		#ifdef CUDA_BACKEND
		
		//TODO

		#else

		for (int z = 0; z < _input_size; z++) {
			float medium = 0;
			for (int k = 0; k < (int)_forward_map[z].size(); k++)
				medium += fabs(_weights.at(z, _forward_map[z][k]));
			medium /= (float)_size;
			for (int k = 0; k < (int)_forward_map[z].size(); k++) {
				if (fabs(_weights.at(z, _forward_map[z][k])) < medium * alpha)
					_forward_map[z].erase(_forward_map[z].begin() + k);
			}
		}

		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::reset_deltas(const double momentum)
	{
		#ifdef CUDA_BACKEND
		
		if (_deltas.size() > 0) CUDA_Tensor_float_scale(_deltas, momentum);

		#else
		
		for (int z = 0; z < _deltas.size(); z++)
			_deltas[z] *= momentum;

		#endif
	}
	
	////////////////////////////////////////////////////////////
	float Partial::pruned_percent()
	{
		int c = 0;
		for (int z = 0; z < _input_size; z++)
			c += _forward_map[z].size();
		return (float)c / (_input_size * _size);
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Partial::get_type() const
	{
		return Operation::Partial;
	}
	
	////////////////////////////////////////////////////////////
	void Partial::print()
	{
		printf("Type: Partial, Size: %d, Input_Size: %d, Connectivity: %f, Weights: %d", _size, _input_size, _connectivity, _size * (_input_size + 1));
	}

} /* namespace lg */
