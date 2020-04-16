// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "CPU_backend.hpp"
#include <math.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	///	UTIL
	////////////////////////////////////////////////////////////
	const float randf() { return (double)rand() / (double)RAND_MAX; }

	////////////////////////////////////////////////////////////
	///	IMPLEMENTATION
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	void convn_forward(float* weights, float* bias, float* inputs, float* outputs, int* out_in_map, int input_width, int input_height, int input_count, int stride, int output_width, int output_height, int filters_count, int filter_area)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void convn_backward(float* weights, float* out_errors, float* errors, int* in_weight_map, int* in_out_map, int input_count, int output_size, int input_width, int input_height, int filter_area, int filters_count)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void convn_accumulate_deltas(float* weights_deltas, float* bias_deltas, float* errors, float* inputs, float* outputs, int* out_in_map, int input_count, int input_width, int input_height, int output_size, int filter_area, int filters_count)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void convn_update_parameters(float* weights, float* bias, float* weights_deltas, float* bias_deltas, int filter_area, int input_count, int filter_count, float learningrate)
	{
		int deltas_index = 0;
		int weights_index = 0;

		for (int f = 0; f < filter_count; f++) {

			//Update weights
			for (int k = 0; k < input_count; k++) {

				//Update filter weights for this input
				for (int w = 0; w < filter_area; w++)
					weights[weights_index++] += weights_deltas[deltas_index++] * learningrate;
			}

			//Update bias
			bias[f] += bias_deltas[f] * learningrate;
		}
	}

	////////////////////////////////////////////////////////////
	void linear_forward(float* weights, float* bias, float* inputs, float* outputs, int input_size, int output_size, bool use_bias, bool accumulate)
	{
		if (use_bias)
		{
			if (!accumulate)
			{
				//Reset outputs with bias
				for (int c = 0; c < output_size; c++)
					outputs[c] = bias[c];
			}
			else
			{
				//Add bias to output
				for (int c = 0; c < output_size; c++)
					outputs[c] += bias[c];
			}
		}
		else
		{
			if (!accumulate)
			{
				for (int c = 0; c < output_size; c++)
					outputs[c] = 0;
			}
		}

		//Compute all inputs
		int weight_index = 0;
		for (int c = 0; c < input_size; c++) {
			if (inputs[c] == 0) continue;
			weight_index = c * output_size;
			for (int k = 0; k < output_size; k++)
				outputs[k] += weights[weight_index++] * inputs[c];
		}
	}

	////////////////////////////////////////////////////////////
	void linear_backward(float* weights, float* out_errors, float* errors, int input_size, int output_size)
	{
		for (int c = 0; c < output_size; c++) {
			if (errors[c] == 0) continue;
			for (int k = 0; k < input_size; k++)
				out_errors[k] += weights[k * output_size + c] * errors[c];
		}
	}

	////////////////////////////////////////////////////////////
	void linear_accumulate_deltas(float* deltas, float* inputs, float* errors, int input_size, int output_size, bool use_bias)
	{
		int d = 0;
		for (int c = 0; c < output_size; c++) {
			for (int k = 0; k <	input_size; k++)
				deltas[d++] += inputs[k] * errors[c];
			if (use_bias) deltas[d++] += errors[c];
			else d++;
		}
	}

	////////////////////////////////////////////////////////////
	void linear_update_parameters(float* weights, float* bias, float* deltas, float learningrate, int input_size, int output_size)
	{
		int d = 0;
		for (int c = 0; c < output_size; c++) {
			for (int k = 0; k <	input_size; k++)
				weights[k * output_size + c] += deltas[d++] * learningrate;
			bias[c] += deltas[d++] * learningrate;
		}
	}
	
	////////////////////////////////////////////////////////////
	void capsules_dense_forward(float* weights, float* bias, float* inputs, float* outputs, float* input_coupling_coeff,
		int input_size, int input_capsule_size, int output_size, int output_capsule_size, int capsule_size, bool use_bias)
	{
		if (use_bias)
		{
				//Reset outputs with bias
				for (int c = 0; c < output_size; c++)
					outputs[c] = bias[c];
		}
		else
		{
				for (int c = 0; c < output_size; c++)
					outputs[c] = 0;
		}

		//Compute all inputs
		int weight_index = 0;
		int coef_id;
		const int coef_stride = input_size / input_capsule_size;
		for (int c = 0; c < input_size; c++) {
			if (inputs[c] == 0) continue;
			weight_index = c * output_size;
			coef_id = c / input_capsule_size;
			for (int k = 0; k < output_size; k++)
				outputs[k] += input_coupling_coeff[k * coef_stride + coef_id] * weights[weight_index++] * inputs[c];
		}
	}

	////////////////////////////////////////////////////////////
	void normalization_forward(float* inputs, float* deviation, float* normalized, float* outputs, float* variance, float* gamma, float* beta, float epsilon, int size)
	{
		//Calculate mean
		double mean = 0;
		for (int c = 0; c < size; c++)
			mean += inputs[c];
		mean /= (double)size;

		//Subtract mean vector to all inputs and calculate variance
		*variance = 0;
		for (int c = 0; c < size; c++) {
			deviation[c] = inputs[c] - mean;
			*variance += deviation[c] * deviation[c];
		}
		*variance /= (double)size;

		//Calculate normalized vector
		for (int c = 0; c < size; c++) {
			normalized[c] = deviation[c] / sqrt(*variance + epsilon);
			outputs[c] = normalized[c] * *gamma + *beta;
		}
	}

	////////////////////////////////////////////////////////////
	void normalization_backward(float* errors, float* out_errors, float* deviation, float* variance, float* gamma, float* beta, float epsilon, int size)
	{
		//Pre-compute some expressions
		float sum_errors = 0.f;
		float sum_errors_dev = 0.f;
		for (int c = 0; c < size; c++) {
			sum_errors += errors[c];
			sum_errors_dev += errors[c] * deviation[c];
		}

		//Calculate output errors
		for (int c = 0; c < size; c++) {
			out_errors[c] = 1.0 / (float)size * *gamma / sqrt(*variance + epsilon) * ((float)size *
					errors[c] - sum_errors - deviation[c] / (*variance + epsilon) * sum_errors_dev);
		}
	}

	////////////////////////////////////////////////////////////
	void normalization_accumulate_deltas(float* errors, float* deviation, float* variance, float* d_gamma, float* d_beta, float epsilon, int size)
	{
		//Calculate beta delta
		for (int c = 0; c < size; c++)
			*d_beta += errors[c];

		//Calculate gamma delta
		for (int c = 0; c < size; c++)
			*d_gamma += deviation[c] * sqrt(*variance + epsilon) * errors[c];
	}

	////////////////////////////////////////////////////////////
	void normalization_update_parameters(float* gamma, float* beta, float* d_gamma, float* d_beta, float momentum, int size, float learningrate)
	{
		*beta += ((double)*d_beta / (double)size) * learningrate;
		*gamma += ((double)*d_gamma / (double)size) * learningrate;
		*d_beta *= momentum;
		*d_gamma *= momentum;
	}

	////////////////////////////////////////////////////////////
	void dropout_forward(const float* inputs, float* outputs, const unsigned int size, const float drop_probability, const bool training)
	{
		if (training == true)
		{
			for (int z = 0; z < size; z++)
				outputs[z] = (randf() < drop_probability) ? 0 : inputs[z];
		}
		else
		{
			for (int z = 0; z < size; z++)
				outputs[z] = inputs[z];
		}
	}

	////////////////////////////////////////////////////////////
	void dropout_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size, const float drop_probability)
	{
		for (int z = 0; z < size; z++)
			out_errors[z] = (outputs[z] == 0) ? 0 :  errors[z] * (1 - drop_probability);
	}

	////////////////////////////////////////////////////////////
	void maxpooling_forward(float* inputs, float* outputs, int* maxbuffer, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier
		int stopX, stopY;
		double max;
		int sx, sy;
		int inputx, inputy;
		float tmp;

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Get input X and Y
					inputx = x * stride;
					inputy = y * stride;

					//Check for bad borders
					stopX = (inputx + filter_size > input_width) ? (inputx + filter_size) - inputx : filter_size;
					stopY = (inputy + filter_size > input_height) ? (inputy + filter_size) - inputy : filter_size;

					//Get max of kernel region
					max = -0x6FFFFFF;
					for (sx = 0; sx < stopX; sx++) {
						for (sy = 0; sy < stopY; sy++) {

							//Store value and check if it's greater, we must do all the calculus
							//because 'data' may be a tensor pointer and have wrong dimensions
							tmp = inputs[f * input_width * input_height + input_width * (inputy + sy) + inputx + sx];
							if (tmp > max) {
								max = tmp;
								maxbuffer[outid] = f * input_width * input_height + input_width * (inputy + sy) + inputx + sx;
							}
						}
					}

					//Update output
					outputs[outid] = max;

					//Next output
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void maxpooling_backward(float* out_errors, float* errors, int* maxbuffer, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Backpropagate errors to each subregion
					out_errors[maxbuffer[outid]] += errors[outid];

					//Update output index
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void averagepooling_forward(float* inputs, float* outputs, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier
		int stopX, stopY;
		int sx, sy;
		int inputx, inputy;

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Get input X and Y
					inputx = x * stride;
					inputy = y * stride;

					//Check for bad borders
					stopX = (inputx + filter_size > input_width) ? (inputx + filter_size) - inputx : filter_size;
					stopY = (inputy + filter_size > input_height) ? (inputy + filter_size) - inputy : filter_size;

					//Get max of kernel region
					outputs[outid] = 0;
					for (sx = 0; sx < stopX; sx++)
						for (sy = 0; sy < stopY; sy++)
							outputs[outid] += inputs[f * input_width * input_height + input_width * (inputy + sy) + inputx + sx];
					outputs[outid] /= (float)stopX * stopY;

					//Next output
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void averagepooling_backward(float* out_errors, float* errors, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier
		int stopX, stopY;
		int sx, sy;
		int inputx, inputy;

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Get input X and Y
					inputx = x * stride;
					inputy = y * stride;

					//Check for bad borders
					stopX = (inputx + filter_size > input_width) ? (inputx + filter_size) - inputx : filter_size;
					stopY = (inputy + filter_size > input_height) ? (inputy + filter_size) - inputy : filter_size;

					//Get max of kernel region
					for (sx = 0; sx < stopX; sx++)
						for (sy = 0; sy < stopY; sy++)
							out_errors[f * input_width * input_height + input_width * (inputy + sy) + inputx + sx] += errors[outid];

					//Update output index
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void relu_forward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++)
			outputs[c] = inputs[c] * (inputs[c] > 0);
	}

	////////////////////////////////////////////////////////////
	void relu_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++)
			out_errors[c] = errors[c] * (outputs[c] > 0);
	}

	////////////////////////////////////////////////////////////
	void tanh_forward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++)
			outputs[c] = tanh(inputs[c]);
	}

	////////////////////////////////////////////////////////////
	void tanh_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++)
			out_errors[c] = (1.f - outputs[c] * outputs[c]) * errors[c];
	}

	////////////////////////////////////////////////////////////
	void sigmoid_forward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++)
			outputs[c] = 1.0 / (1.0 + exp(-inputs[c]));
	}

	////////////////////////////////////////////////////////////
	void sigmoid_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++)
			out_errors[c] = outputs[c] * (1.f - outputs[c]) * errors[c];
	}

	static const double _alpha = 1.6732632423543772;
	static const double _scale = 1.0507009873554804;

	////////////////////////////////////////////////////////////
	void selu_forward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++) {
			if (inputs[c] >= 0.0) outputs[c] = _scale * inputs[c];
			else outputs[c] = _scale * (_alpha * exp(inputs[c]) - _alpha);
		}
	}

	////////////////////////////////////////////////////////////
	void selu_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int c = 0; c < size; c++) {
			if (outputs[c] > 0.0) out_errors[c] = _scale * errors[c];
			else out_errors[c] = errors[c] * (outputs[c] + _scale * _alpha);
		}
	}

	////////////////////////////////////////////////////////////
	void softmax_forward(float* inputs, float* outputs, float scale, int size, float epsilon)
	{
		//Calculate sum of all inputs
		double sum = 0;
		for (int c = 0; c < size; c++) {
			outputs[c] = exp(inputs[c] * scale);
			sum += outputs[c];
		}

		//Calculate outputs
		for (int c = 0; c < size; c++)
			outputs[c] = outputs[c] / (sum + epsilon);
	}

	////////////////////////////////////////////////////////////
	void softmax_backward(float* errors, float* out_errors, float* outputs, int size)
	{
		for (int c = 0; c < size; c++)
			out_errors[c] = outputs[c] * (1.f - outputs[c]) * errors[c];
	}
	
	////////////////////////////////////////////////////////////
	void capsule_squashing_forward(float* inputs, float* outputs, int size)
	{
		double length = 0.f;
		for (int c = 0; c < size; c++)
			length += inputs[c] * inputs[c];
		length = sqrt(length);
		
		const float multiplier = (length * length) / (1.f + length * length);
		for (int c = 0; c < size; c++)
			outputs[c] = multiplier * (inputs[c] / length);
	}

	////////////////////////////////////////////////////////////
	void capsule_squashing_backward(float* errors, float* out_errors, float* outputs, int size)
	{

	}

	////////////////////////////////////////////////////////////
	void gradient_clipping(float* deltas, int size, const float clipping_deviation)
	{
		for (int c = 0; c < size; c++) {
			if (deltas[c] > clipping_deviation) deltas[c] = clipping_deviation;
			else if(deltas[c] < -clipping_deviation) deltas[c] = -clipping_deviation;
		}
	}

	////////////////////////////////////////////////////////////
	void l1_regularization(float* weights, const float l1_factor, const float learningrate, int size)
	{
		for (int c = 0; c < size; c++)
			weights[c] += (weights[c] > 0 ? -1.f : 1.f) * l1_factor * learningrate;
	}

	////////////////////////////////////////////////////////////
	void l2_regularization(float* weights, const float l2_factor, const float learningrate, int size)
	{
		for (int c = 0; c < size; c++)
			weights[c] += (0 - weights[c]) * l2_factor * learningrate;
	}

} /* namespace lg */
