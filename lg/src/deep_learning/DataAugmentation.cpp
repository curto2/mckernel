// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "DataAugmentation.hpp"
#include "../util/ensure.hpp"
#include "../util/Util.hpp"
#ifdef CUDA_BACKEND
#include "../deeplearning/CUDA_backend.hpp"
#endif


#define AUGMENTATION_MAX_GENERABLE_IMAGES 1000000

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	////////////////////////////////////////////////////////////
	///	NAMESPACE AUGMENTATION
	////////////////////////////////////////////////////////////
	namespace augmentation
	{
		
		#ifdef CUDA_BACKEND
		
		////////////////////////////////////////////////////////////
		void translate(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels, int tx, int ty)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);
			
			lg::CUDA_Tensor_float working_buf(t.width(), t.height(), t.depth());
			lg::cuda::image_translate(t.pointer(), working_buf.pointer(), image_width, image_height, image_channels, tx,ty);
			t.copy(working_buf);
		}
		
		////////////////////////////////////////////////////////////
		void rotate(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels, float degrees)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);
			
			lg::CUDA_Tensor_float working_buf(t.width(), t.height(), t.depth());
			lg::cuda::image_rotate(t.pointer(), working_buf.pointer(), image_width, image_height, image_channels, degrees);
			t.copy(working_buf);
		}
		
		////////////////////////////////////////////////////////////
		void noise(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels, float noise)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);
			lg::cuda::image_add_noise(t.pointer(), image_width, image_height, image_channels, noise);
		}
		
		////////////////////////////////////////////////////////////
		void vflip(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);
			lg::cuda::image_vertical_flip(t.pointer(), image_width, image_height, image_channels);
		}
		
		////////////////////////////////////////////////////////////
		void hflip(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);
			lg::cuda::image_horizontal_flip(t.pointer(), image_width, image_height, image_channels);
		}
		
		////////////////////////////////////////////////////////////
		void scaling(CUDA_Tensor_float& t, int image_width, int image_height, int image_channels, float scale_factor)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);
			
			lg::CUDA_Tensor_float working_buf(t.width(), t.height(), t.depth());
			lg::cuda::image_scale(t.pointer(), working_buf.pointer(), image_width, image_height, image_channels, scale_factor);
			t.copy(working_buf);
		}
		

		#else
		
		////////////////////////////////////////////////////////////
		void tensor_copy(const Tensor_float& src, Tensor_float& dst)
		{
			ensure(dst.size() <= src.size());
			for (int z = 0; z < src.size(); z++)
				dst[z] = src[z];
		}

		////////////////////////////////////////////////////////////
		void hflip(Tensor_float& t, int width, int height, int channels)
		{
			//Check errors
			ensure(t.size() == width * height * channels);
			
			float tmp;
			for (int c = 0; c < channels; c++) {
				for (int x = 0; x < width/2; x++) {
					for (int y = 0; y < height; y++) {
						tmp = t[c * width * height + y * width + x];
						t[c * width * height + y * width + x] = t[c * width * height + y * width + (width - 1 - x)];
						t[c * width * height + y * width + (width - 1 - x)] = tmp;
					}
				}
			}
		}
		
		////////////////////////////////////////////////////////////
		void vflip(Tensor_float& t, int width, int height, int channels)
		{
			float tmp;
			for (int c = 0; c < channels; c++) {
				for (int x = 0; x < width; x++) {
					for (int y = 0; y < height/2; y++) {
						tmp = t[c * width * height + y * width + x];
						t[c * width * height + y * width + x] = t[c * width * height + (height -1 - y) * width + x];
						t[c * width * height + (height -1 - y) * width + x] = tmp;
					}
				}
			}
		}
		
		////////////////////////////////////////////////////////////
		void rotate(Tensor_float& t, int image_width, int image_height, int image_channels, float degrees)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);

			//Allocate temporary buffer
			Tensor_float temp(t.size());
			temp.fill(0);

			//Compute constants
			const float angle = degrees * 3.14159f/180.0;
			const float a = cos(angle);
			const float b = sin(angle);
			const int wh = image_width/2.f;
			const int hh = image_height/2.f;
			const int xoffset = wh - (wh * a - hh * b);
			const int yoffset = hh - (wh * b + hh * a);

			//Compute rotation in temporary buffer
			// x' = x * cos(a) + y * sin(a) + xoffset
			// y' = y * cos(a) - x * sin(a) + yoffset
			int nx, ny;
			for (int x = 0; x < image_width; x++) {
				for (int y = 0; y < image_height; y++) {
					nx = x * a - y * b + xoffset;
					ny = x * b + y * a + yoffset;
					if (nx < 0 || nx >= image_width) continue;
					if (ny < 0 || ny >= image_height) continue;
					for (int c = 0; c < image_channels; c++) {
						temp[(y * image_width + x) + c * image_width * image_height] = t[(ny * image_width + nx) + c * image_width * image_height];
					}
				}
			}

			//Copy result to final tensor
			for (int z = 0; z < t.size(); z++) t[z] = temp[z];
		}
		
		////////////////////////////////////////////////////////////
		void translate(Tensor_float& t, int image_width, int image_height, int image_channels, int tx, int ty)
		{
			//Check errors
			ensure(t.size() == image_width * image_height * image_channels);

			//Calculate result in temporary tensor
			Tensor_float temp(t.size());
			int tpos;
			int coffset;
			for (int c = 0; c < image_channels; c++) {
				for (int x = 0; x < image_width; x++) {
					for (int y = 0; y < image_height; y++) {
						tpos = (y - ty) * image_width + (x - tx);
						coffset = image_width * image_height * c;
						if (y - ty < 0 || y - ty >= image_height || x - tx < 0 || x - tx >= image_width) temp[coffset + y * image_width + x] = 0;
						else temp[coffset + y * image_width + x] = t[coffset + tpos];
					}
				}
			}
			
			//Copy result to final tensor
			for (int z = 0; z < t.size(); z++) t[z] = temp[z];
		}

		////////////////////////////////////////////////////////////
		void noise(Tensor_float& t, int image_width, int image_height, int image_channels, float noise)
		{
			ensure(t.size() == image_width * image_height * image_channels);
			for (int z = 0; z < image_width * image_height; z++) {
				if (lg::util::randf() < noise)
					for (int c = 0; c < image_channels; c++) 
						t[image_width * image_height * c + z] = 1 - t[image_width * image_height * c + z];
			}
		}
	
		////////////////////////////////////////////////////////////
		void scaling(Tensor_float& t, int width, int height, int channel, float scale_factor)
		{
			//Check errors
			ensure(t.size() == width * height * channel);
			
			const int cx = width / 2; //center of scaling
			const int cy = height / 2;
			const float scale = 1.f / scale_factor;
			Tensor_float tmp(width, height, channel);
			tmp.fill(0);
			int nx, ny;
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					nx = cx + (x - cx) * scale;
					ny = cy + (y - cy) * scale;
					if (nx < 0 || nx >= width) continue;
					if (ny < 0 || ny >= height) continue;
					for (int c = 0; c < channel; c++) {
						tmp[(y * width + x) * channel + c] = t[(ny * width + nx) * channel + c];
					}
				}
			}
			
			//Copy result to final tensor
			for (int z = 0; z < t.size(); z++) t[z] = tmp[z];
		}

	#endif

	} /* namespace augmentation */
	
} /* namespace lg */
