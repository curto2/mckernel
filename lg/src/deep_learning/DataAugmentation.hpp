// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef DATAUGMENTATION_HPP
#define DATAUGMENTATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../util/Tensor.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "../util/CUDA_Tensor.hpp"
#endif
#include <vector>
#include <string>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	////////////////////////////////////////////////////////////
	///	NAMESPACE AUGMENTATION
	////////////////////////////////////////////////////////////
	namespace augmentation
	{
		
		#ifdef CUDA_BACKEND

		void translate(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels, int tx, int ty);
		void rotate(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels, float degrees);
		void noise(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels, float noise);
		void vflip(CUDA_Tensor_float& t, int width, int height, int channels);
		void hflip(CUDA_Tensor_float& t, int width, int height, int channels);
		void scaling(CUDA_Tensor_float& t, int width, int height, int channel, float scale_factor);

		#else

		void translate(Tensor_float& t, int image_width, int image_height, int image_channels, int tx, int ty);
		void rotate(Tensor_float& t, int image_width, int image_height, int image_channels, float degrees);
		void noise(Tensor_float& t, int image_width, int image_height, int image_channels, float noise);
		void vflip(Tensor_float& t, int width, int height, int channels);
		void hflip(Tensor_float& t, int width, int height, int channels);
		void scaling(Tensor_float& t, int width, int height, int channel, float scale_factor);
		
		#endif

	} /* namespace augmentation */
	
} /* namespace ai */

#endif /* end of include guard: DATAUGMENTATION_HPP */

