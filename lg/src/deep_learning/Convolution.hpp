// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include "Operation.hpp"
#include "../util/Point.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	class Convolution : public Operation
	{
		public:
			Convolution(const int filter_size, const int filter_count, const int stride,
						const int padding = 0, const float gradient_clipping = 0, const float l1_regularization = 0.f,
						const float l2_regularization = 0.f);
			Convolution(const lg::Point filter_size, const int filter_count,
						const int stride, const int padding = 0, const float gradient_clipping = 0, const float l1_regularization = 0.f,
						const float l2_regularization = 0.f);
			Convolution(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void initialize(const int input_width, const int input_height, const int input_count);
			void save(lg::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			void reset_deltas(const double momentum);
			void setFixedParameters(const bool fixedparameters);
			void saveParameters(std::string filepath);
			void loadParameters(std::string filepath);
			void print();
			const Operation::Type get_type() const;
			
			static std::shared_ptr<Operation> make(const int filter_size, const int filter_count,
						const int stride, const int padding = 0, const float gradient_clipping = 0,
						const float l1_regularization = 0.f, const float l2_regularization = 0.f);
			static std::shared_ptr<Operation> make(const lg::Point filter_size, const int filter_count,
						const int stride, const int padding = 0, const float gradient_clipping = 0, 
						const float l1_regularization = 0.f, const float l2_regularization = 0.f);
		
			#ifdef CUDA_BACKEND
			
			void run(const CUDA_Tensor_float &input, const bool training);
			void backprop(CUDA_Tensor_float &out_errors);
			void accumulate_deltas(const CUDA_Tensor_float &input);
			
			lg::cudnn::Convolution cudaconv;
			CUDA_Tensor_float _weights;
			CUDA_Tensor_float _bias;
			CUDA_Tensor_float _bias_deltas;
			CUDA_Tensor_float _weights_deltas;
			CUDA_Tensor_float _workspace;
			CUDA_Tensor_int _in_weights_map;
			CUDA_Tensor_int _in_out_map;
			CUDA_Tensor_int _out_in_map;
			
			#else
			
			static void gradient_check();
			void run(const Tensor_float &input, const bool training);
			void backprop(Tensor_float &out_errors);
			void accumulate_deltas(const Tensor_float &input);
			
			Tensor_float _weights;
			Tensor_float _bias;
			Tensor_float _bias_deltas;
			Tensor_float _weights_deltas;
			
			#endif
			bool _fixed_parameters;
			int _input_width;
			int _input_height;
			int _input_count;
			int _filter_width;
			int _filter_height;
			int _filter_count;
			int _stride;
			int _output_width;
			int _output_height;
			int _output_size;
			int _input_size;
			int _padding;
			float _gradient_clipping;
			float _l1_regularization;
			float _l2_regularization;
			std::vector<std::vector< int >> _convmap;
	};

} /* namespace lg */

#endif /* end of include guard: CONVOLUTION_HPP */

