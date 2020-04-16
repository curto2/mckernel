// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "WeightRegularization.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
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
			
		////////////////////////////////////////////////////////////		
		void l1_regularization(CUDA_Tensor_float& weights, const float l1_factor, const float learningrate)
		{
			cuda::l1_regularization(weights.pointer(), l1_factor, learningrate, weights.size());
		}

		////////////////////////////////////////////////////////////		
		void l2_regularization(CUDA_Tensor_float& weights, const float l2_factor, const float learningrate)
		{
			cuda::l2_regularization(weights.pointer(), l2_factor, learningrate, weights.size());
		}

		#else
		
		////////////////////////////////////////////////////////////
		void l1_regularization(Tensor_float& weights, const float l1_factor, const float learningrate)
		{
			for (int c = 0; c < weights.size(); c++)
				weights[c] += (weights[c] > 0 ? -1.f : 1.f) * l1_factor * learningrate;
		}
				
		////////////////////////////////////////////////////////////
		void l2_regularization(Tensor_float& weights, const float l2_factor, const float learningrate)
		{
			for (int c = 0; c < weights.size(); c++)
				weights[c] += (0 - weights[c]) * l2_factor * learningrate;
		}
		
		#endif
		
	} /* namespace wrn */
	
} /* namespace lg */
