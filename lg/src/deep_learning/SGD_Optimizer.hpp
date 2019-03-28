// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef SGD_Optimizer_HPP
#define SGD_Optimizer_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Optimizer.hpp"
#include "../util/Macros.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class SGD_Optimizer : public Optimizer
	{
		public:
			SGD_Optimizer();
			SGD_Optimizer(const int batch_size, const double learningrate, const double momentum,
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
			#endif
	};
} /* namespace ai */

#endif /* end of include guard: SGD_Optimizer_HPP */

