// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "visualization.hpp"
#include "Bitmap.hpp"
#include <assert.h>
#include <algorithm>
#include <math.h>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	////////////////////////////////////////////////////////////
	///	NAMESPACE VISUALIZATION
	////////////////////////////////////////////////////////////
	namespace visualization
	{
		
		////////////////////////////////////////////////////////////
		///	HIDDEN FUNCTIONS
		////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////
		/// \brief	Find the nearest perfect square of a number	
		///
		////////////////////////////////////////////////////////////
		int findminsquare(int num)
		{
			return (int)(sqrt((float)num) + 0.99);
		}

		////////////////////////////////////////////////////////////
		/// \brief	Find the minimum and maximum value in a vector
		/// of numbers
		///
		////////////////////////////////////////////////////////////
		void findvecrange(float* data, const int size, double *min, double *max)
		{
			if (min != NULL) *min = 0xFFFFFF;
			if (max != NULL) *max = -0xFFFFFF;
			for (int c = 0; c < (int)size; c++) {
				if (min != NULL && data[c] < *min) *min = data[c];
				if (max != NULL && data[c] > *max) *max = data[c];
			}
		}
		
		////////////////////////////////////////////////////////////
		///	PUBLIC FUNCTIONS
		////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////
		void save_vec(std::string path, const Tensor_float& vector)
		{
			const int img_size = findminsquare(vector.size());
			double min, max;
			
			//Find min and max for normalization
			findvecrange(vector.pointer(), vector.size(), &min, &max);
			
			//Store and normalize data into a bitmap
			Bitmap bm(img_size, img_size, Bitmap::MONO, 0x000000);
			for (int c = 0; c < (int)vector.size(); c++)
				bm.m_data[c] = ((vector[c] - min) / (max - min)) * 255;

			//Save bitmap
			bm.save(path);
		}
		
		////////////////////////////////////////////////////////////
		void save_multiple_vec(std::string path, const Tensor_float vector)
		{
			//Get final bitmap dimensions
			int img_width = 0;
			int img_height = 0;
			for (int c = 0; c < (int)vector.height(); c++) {
				const int block_size = findminsquare(vector.width());
				img_width += block_size + 1;
				if (block_size > img_height) img_height = block_size;
			}
				
			//Create final bitmap
			Bitmap img(img_width, img_height, Bitmap::MONO, 0x000000);
			
			int pos = 0;
			for (int z = 0; z < vector.height(); z++) {
				const int img_size = findminsquare(vector.width());
				double min, max;

				//Find min and max for normalization
				const lg::Tensor_float tmp = vector.ptr(0, z);
				findvecrange(tmp.pointer(), vector.width(), &min, &max);

				//Store and normalize data into a bitmap
				Bitmap bm(img_size, img_size, Bitmap::MONO, 0x000000);
				for (int c = 0; c < (int)vector.width(); c++)
					bm.m_data[c] = ((vector.at(z, c) - min) / (max - min)) * 255;
				
				//Copy block to final bitmap
				bm.copyToRegion(img, 0, 0, img_size, img_size, pos, 0, img_size, img_size);
				pos += img_size + 1;
			}

			img.save(path);
		}
		
		////////////////////////////////////////////////////////////
		void save_multiple_vec(std::string path, const Tensor_float vector, int table_width, int table_height)
		{
			assert(vector.size() <= table_width * table_height);

			//Get final bitmap dimensions
			int img_width = findminsquare(vector.height());
			int img_height = img_width;
				
			//Create final bitmap
			Bitmap img(table_width * img_width, table_height * img_height, Bitmap::MONO, 0x000000);
					
			for (int z = 0; z < (int)vector.size(); z++) {
				double min, max;

				//Find min and max for normalization
				const lg::Tensor_float tmp = vector.ptr(0, z);
				findvecrange(tmp.pointer(), vector.width(), &min, &max);

				//Store and normalize data into a bitmap
				Bitmap bm(img_width, img_height, Bitmap::MONO, 0x000000);
				for (int c = 0; c < (int)vector.width(); c++)
					bm.m_data[c] = ((vector.at(z, c) - min) / (max - min)) * 255;
				
				//Copy block to final bitmap
				bm.copyToRegion(img, 0, 0, img_width, img_height, (z % table_width) * img_width,
								(z / table_width) * img_height, img_width, img_height);
			}

			img.save(path);
		}

		////////////////////////////////////////////////////////////
		void save_vec(std::string path, const Tensor_float& vector, int width, int height)
		{
			//Check if the size of the vector is correct
			assert(vector.size() <= width * height);

			//Find min and max for normalization
			double min, max;
			findvecrange(vector.pointer(), vector.size(), &min, &max);
			
			//Store and normalize data into a bitmap
			Bitmap bm(width, height, Bitmap::MONO, 0x000000);
			for (int c = 0; c < (int)vector.size(); c++)
				bm.m_data[c] = ((vector[c] - min) / (max - min)) * 255;

			//Save bitmap
			bm.save(path);
		}

	} //namespace visualization

} //namespace lg
