// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef CPU_BACKEND_HPP
#define CPU_BACKEND_HPP

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	void convn_forward(float* weights, float* bias, float* inputs, float* outputs, int* out_in_map, int input_width, int input_height, int input_count, int stride, int output_width, int output_height, int filters_count, int filter_area);
	void convn_backward(float* weights, float* out_errors, float* errors, int* in_weight_map, int* in_out_map, int input_count, int output_size, int input_width, int input_height, int filter_area, int filters_count);
	void convn_accumulate_deltas(float* weights_deltas, float* bias_deltas, float* errors, float* inputs, float* outputs, int* out_in_map, int input_count, int input_width, int input_height, int output_size, int filter_area, int filters_count);
	void convn_update_parameters(float* weights, float* bias, float* weights_deltas, float* bias_deltas, int filter_area, int input_count, int filter_count, float learningrate);
	void linear_forward(float* weights, float* bias, float* inputs, float* outputs, int input_size, int output_size, bool use_bias, bool accumulate);
	void linear_backward(float* weights, float* out_errors, float* errors, int input_size, int output_size);
	void linear_accumulate_deltas(float* deltas, float* inputs, float* errors, int input_size, int output_size, bool use_bias);
	void linear_update_parameters(float* weights, float* bias, float* deltas, float learningrate, int input_size, int output_size);
	void capsules_dense_forward(float* weights, float* bias, float* inputs, float* outputs, float* input_coupling_coeff,
		int input_size, int input_capsule_size, int output_size, int output_capsule_size, int capsule_size, bool use_bias);
	void normalization_forward(float* inputs, float* deviation, float* normalized, float* outputs, float* variance, float* gamma, float* beta, float epsilon, int size);
	void normalization_backward(float* errors, float* out_errors, float* deviation, float* variance, float* gamma, float* beta, float epsilon, int size);
	void normalization_accumulate_deltas(float* errors, float* deviation, float* variance, float* d_gamma, float* d_beta, float epsilon, int size);
	void normalization_update_parameters(float* gamma, float* beta, float* d_gamma, float* d_beta, float momentum, int size, float learningrate);
	void dropout_forward(const float* inputs, float* outputs, const unsigned int size, const float drop_probability, const bool training);
	void dropout_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size, const float drop_probability);
	void maxpooling_forward(float* inputs, float* outputs, int* maxbuffer, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height);
	void maxpooling_backward(float* out_errors, float* errors, int* maxbuffer, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height);
	void averagepooling_forward(float* inputs, float* outputs, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height);
	void averagepooling_backward(float* out_errors, float* errors, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height);
	void relu_forward(const float* inputs, float* outputs, const unsigned int size);
	void relu_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size);
	void tanh_forward(const float* inputs, float* outputs, const unsigned int size);
	void tanh_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size);
	void sigmoid_forward(const float* inputs, float* outputs, const unsigned int size);
	void sigmoid_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size);
	void selu_forward(const float* inputs, float* outputs, const unsigned int size);
	void selu_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size);
	void softmax_forward(float* inputs, float* outputs, float scale, int size, float epsilon);
	void softmax_backward(float* errors, float* out_errors, float* outputs, int size);
	void capsule_squashing_forward(float* inputs, float* outputs, int size);
	void capsule_squashing_backward(float* errors, float* out_errors, float* outputs, int size);	
	void gradient_clipping(float* deltas, int size, const float clipping_deviation);
	void l1_regularization(float* weights, const float l1_factor, const float learningrate, int size);
	void l2_regularization(float* weights, const float l2_factor, const float learningrate, int size);

} /* namespace lg */

#endif /* end of include guard: CPU_BACKEND_HPP */

