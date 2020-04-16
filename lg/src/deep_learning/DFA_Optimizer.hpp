// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef OPTIMIZERDFA_HPP
#define OPTIMIZERDFA_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Optimizer.hpp"
#include "../util/Macros.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	class DFA_Optimizer : public Optimizer
	{
		public:
			DFA_Optimizer();
			DFA_Optimizer(const int batch_size, const double learningrate, const double momentum,
					const Cost::CostType cost_function = Cost::SquaredError);
			#ifdef CUDA_BACKEND
			void fit(Neural_Network& net, CUDA_Tensor_float &inputs, CUDA_Tensor_float &targets);
			#else
			void fit(Neural_Network& net, Tensor_float &inputs, Tensor_float &targets);
			#endif

		private:
			int _batch_size;
			int _current_sample;
			#ifdef CUDA_BACKEND
			CUDA_Tensor_float _targets;
			CUDA_Tensor_float _feedback_weights;
			CUDA_Tensor_float _feedback_errors;
			#else
			Tensor_float _feedback_weights;
			Tensor_float _feedback_errors;
			#endif
	};
} /* namespace lg */

#endif /* end of include guard: OPTIMIZERDFA_HPP */

