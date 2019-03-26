// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

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
	class Autoencoder : public Operation
	{
		public:
			Autoencoder();
			Autoencoder(const int size, const float noise);
			Autoencoder(ai::IOData& data);
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
			const float getPredictionError();
			const Operation::Type get_type() const;
			void print();
			
			static std::shared_ptr<Operation> make(const int size, const float noise);
			

			#ifdef CUDA_BACKEND

			void run(const CUDA_Tensor_float& input, bool accumulate, bool training);
			void backprop(CUDA_Tensor_float& out_errors);
			void accumulate_deltas(const CUDA_Tensor_float& input);
			
			ai::CUDA_Tensor_float _weights, _bias, _w_deltas, _b_deltas, _prediction, _prediction_error, _hidden_errors, _noise_mask;
			ai::CUDA_Tensor_int _activations;

			#else
			
			void run(const Tensor_float input, bool accumulate, bool training);
			void backprop(Tensor_float out_errors);
			void accumulate_deltas(const Tensor_float input);
			
			ai::Tensor_float _weights, _bias, _w_deltas, _b_deltas, _prediction, _prediction_error, _hidden_errors, _noise_mask;
			ai::Tensor_int _activations;

			#endif
			float _error;
			float _noise;
			float _learningrate;
			bool _fixed_parameters;
			int _input_size;
	};

} /* namespace ai */

#endif /* end of include guard: AUTOENCODER_HPP */

