// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Cost.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "../util/CUDA_Tensor.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{	
	
	class Neural_Network;

	class Optimizer
	{
		public:
			virtual void fit(Neural_Network& net, Tensor_float &inputs, Tensor_float &targets);
			#ifdef CUDA_BACKEND
			virtual void fit(Neural_Network& net, CUDA_Tensor_float &inputs, CUDA_Tensor_float &targets);
			#endif
			void setLearningrate(const float learningrate);
			void setMomentum(const float momentum);
			const float getLearningrate() const;
			const float getMomentum() const;
			const float getError() const;

		protected:
			float _learningrate; 
			float _momentum;
			float _error;
			Cost _costfunction;
	};

} /* namespace lg */

#endif /* end of include guard: OPTIMIZER_HPP */

