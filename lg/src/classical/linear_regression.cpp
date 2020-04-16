// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "linear_regression.hpp"
#include "../util/ensure.hpp"
#include <algorithm>
#include <math.h>
#include <fstream>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	////////////////////////////////////////////////////////////
	linear_regression::linear_regression(const unsigned int input_size, const unsigned int output_size)
	{
		_input_size = input_size;
		_output_size = output_size;
		_weights.setshape(output_size, input_size);
		_weights.fill(0.0, 1.f / sqrt(input_size));
		_bias.setshape(output_size);
		_bias.fill(0.0, 1.f / sqrt(input_size));
		_outputs.setshape(output_size);
		_outputs.fill(0);
		_errors.setshape(output_size);
		_errors.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	linear_regression::linear_regression(const std::string filepath)
	{
		std::ifstream file(filepath, std::ios::binary);
		ensure(file && "can't open the file for reading!");
		file.read(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.read(reinterpret_cast<char*>(&_output_size), sizeof(_output_size));
		_weights.setshape(_output_size, _input_size);
		_bias.setshape(_output_size);
		_outputs.setshape(_output_size);
		_outputs.fill(0);
		_errors.setshape(_output_size);
		_errors.fill(0);
		file.read(reinterpret_cast<char*>(&_weights[0]), sizeof(float) * _input_size * _output_size);
		file.read(reinterpret_cast<char*>(&_bias[0]), sizeof(float) * _output_size);
	}
	
	////////////////////////////////////////////////////////////
	void linear_regression::save(const std::string filepath)
	{
		std::ofstream file(filepath, std::ios::binary);
		ensure(file && "can't open the file for writing!");
		file.write(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.write(reinterpret_cast<char*>(&_output_size), sizeof(_output_size));
		file.write(reinterpret_cast<char*>(&_weights[0]), sizeof(float) * _input_size * _output_size);
		file.write(reinterpret_cast<char*>(&_bias[0]), sizeof(float) * _output_size);
	}

	////////////////////////////////////////////////////////////
	const Tensor_float& linear_regression::predict(const Tensor_float input)
	{
		ensure(input.width() == (int)_input_size);
		ensure(input.height() == 1 && input.depth() == 1);
		
		for (int k = 0; k < (int)_output_size; k++)
			_outputs[k] = _bias[k];	
		
		for (unsigned int c = 0; c < _input_size; c++) {
			if (input[c] == 0) continue;
			for (unsigned int k = 0; k < _output_size; k++) {
				_outputs[k] += _weights.at(c, k) * input[c];
			}
		}

		return _outputs;
	}
	
	////////////////////////////////////////////////////////////
	void linear_regression::fit(Tensor_float inputs, const Tensor_float targets, const float starting_learningrate,
		const unsigned int epochs, const bool verbose)
	{
		ensure(inputs.width() == (int)_input_size);
		ensure(inputs.height() > 1 && inputs.depth() == 1);
		ensure(targets.width() == (int)_output_size);
		ensure(targets.height() == inputs.height());


		float learning_rate = starting_learningrate;
		double medium_error = 0;
		for (unsigned int e = 0; e < epochs; e++) {	
			
			medium_error = 0;
		
			std::vector<int> shuffle_idx(inputs.height());
			for (int c = 0; c < (int)shuffle_idx.size(); c++) shuffle_idx[c] = c;
			std::random_shuffle(shuffle_idx.begin(), shuffle_idx.end());

			for (int c = 0; c < (int)inputs.height(); c++) {
				
				//Compute learning rate
				learning_rate = starting_learningrate * (1.f - (double)(e * inputs.height() + c) / (double)(epochs * inputs.height()));
				
				int indx = shuffle_idx[c];
				
				//Calculate output
				predict(inputs.ptr(0, indx));

				//Calculate prediction errors
				for (unsigned int z = 0; z < _output_size; z++) {
					_errors[z] = targets.at(indx, z) - _outputs[z];
					medium_error += fabs(_errors[z]);
				}

				//Update weights
				for (unsigned int z = 0; z < _output_size; z++) {
					_bias[z] += _errors[z] * learning_rate;
					for (unsigned int k = 0; k < _input_size; k++)
						_weights.at(k, z) += _errors[z] * inputs.at(indx, k) * learning_rate; 
				}
			}
			if (verbose) printf("Epoch: %d Mean Error: %f\n", e, medium_error / (double)inputs.height());
			learning_rate *= 0.7;
		}
	}

	////////////////////////////////////////////////////////////
	const Tensor_float& linear_regression::get_output()
	{
		return _outputs;
	}

} /* namespace lg */
