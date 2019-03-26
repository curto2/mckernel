// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef LINEAR_HPP
#define LINEAR_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include "Operation.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	class Linear : public Operation
	{
		public:
			Linear();
			Linear(const int size, bool use_bias = true, const float gradient_clipping = 0, float l1_regularization = 0, float l2_regularization = 0);
			Linear(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void initialize(int input_size);
			void save(ai::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			void reset_deltas(const double momentum);
			void reset_outputs();
			void setFixedParameters(const bool fixedparameters);
			const Operation::Type get_type() const;
			void print();
			
			static std::shared_ptr<Operation> make(const int size, bool use_bias = true,
				const float gradient_clipping = 0, float l1_regularization = 0, float l2_regularization = 0);
			

			#ifdef CUDA_BACKEND

			void run(const CUDA_Tensor_float& input, bool accumulate);
			void backprop(CUDA_Tensor_float& out_errors);
			void accumulate_deltas(const CUDA_Tensor_float& input);

			CUDA_Tensor_float _weights;
			CUDA_Tensor_float _bias;
			CUDA_Tensor_float _deltas;
			CUDA_Tensor_int _sparse_indices;
			CUDA_Tensor_int _sparse_indices_tmp;
			CUDA_Tensor_int _sparse_indices_count;

			#else
			
			static void gradient_check();
			void run(const Tensor_float input, bool accumulate);
			void backprop(Tensor_float out_errors);
			void accumulate_deltas(const Tensor_float input);
			
			Tensor_float _weights;
			Tensor_float _bias;
			Tensor_float _deltas;
			
			#endif
			bool _fixed_parameters;
			int _input_size;
			bool _use_bias;
			float _gradient_clipping;
			float _l1_regularization;
			float _l2_regularization;
	};

} /* namespace ai */

#endif /* end of include guard: LINEAR_HPP */

