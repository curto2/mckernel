// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef WEIGHTREGULARIZATION_HPP
#define WEIGHTREGULARIZATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../util/Macros.hpp"
#include "../util/Tensor.hpp"
#ifdef CUDA_BACKEND
#include "../util/CUDA_Tensor.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	////////////////////////////////////////////////////////////
	///	NAMESPACE WEIGHTREGULARIZATION
	////////////////////////////////////////////////////////////
	namespace weightreg
	{
		
		#ifdef CUDA_BACKEND
		
		void l1_regularization(CUDA_Tensor_float& weights, const float l1_factor, const float learningrate);
		void l2_regularization(CUDA_Tensor_float& weights, const float l2_factor, const float learningrate);

		#else

		void l1_regularization(Tensor_float& weights, const float l1_factor, const float learningrate);
		void l2_regularization(Tensor_float& weights, const float l2_factor, const float learningrate);
		
		#endif

	} /* namespace weightregularization */

} /* namespace ai */

#endif /* end of include guard: WEIGHTREGULARIZATION_HPP */

