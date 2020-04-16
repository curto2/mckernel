// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

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
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	
	////////////////////////////////////////////////////////////
	///	NAMESPACE WEIGHTREGULARIZATION
	////////////////////////////////////////////////////////////
	namespace wrn
	{
		
		#ifdef CUDA_BACKEND
		
		void l1_regularization(CUDA_Tensor_float& weights, const float l1_factor, const float learningrate);
		void l2_regularization(CUDA_Tensor_float& weights, const float l2_factor, const float learningrate);

		#else

		void l1_regularization(Tensor_float& weights, const float l1_factor, const float learningrate);
		void l2_regularization(Tensor_float& weights, const float l2_factor, const float learningrate);
		
		#endif

	} /* namespace wrn */

} /* namespace lg */

#endif /* end of include guard: WEIGHTREGULARIZATION_HPP */

