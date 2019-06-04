// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef TENSORCUDA_HPP
#define TENSORCUDA_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Tensor.hpp"
#include "IOData.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	template < typename T >
	class CUDA_Tensor
	{
		public:
			CUDA_Tensor();
			CUDA_Tensor(const CUDA_Tensor<T>& t);
			CUDA_Tensor(int width);
			CUDA_Tensor(int width, int height);
			CUDA_Tensor(int width, int height, int depth);
			~CUDA_Tensor();

			void load(lg::IOData& data, std::string dataname);
			void load(std::ifstream& file);
			void save(lg::IOData& data, std::string dataname);
			void save(std::ofstream& file);
            void setshape(const int width);
            void setshape(const int width, const int height);
            void setshape(const int width, const int height, const int depth);
            void setshape(Tensor<T>& host_tensor);
            void point(const CUDA_Tensor<T>& t);
            void point(const CUDA_Tensor<T>& t, const unsigned int offset_d);
            void point(const CUDA_Tensor<T>& t, const unsigned int offset_d, const unsigned int offset_y);
			void clear();
			void copy(const CUDA_Tensor<T>& tensor);
			void copyToHost(T *arr, int size) const;
			void copyToDevice(const T *arr, int size);
			void fill(T val);
			void fill(float mean, float dev);
			CUDA_Tensor<T> ptr(const int d);
			CUDA_Tensor<T> ptr(const int d, const int y);
			inline T* pointer() const { return _data;}
			inline const int size() const { return _size; }
			inline const int width() const { return _width; }
			inline const int height() const { return _height; }
			inline const int depth() const { return _depth; }


		private:
			T* _data = NULL;
			int _size = 0;
			int _width = 0, _height = 0, _depth = 0;
			bool _owner = false;
	};
	
	//Shortcut
	typedef CUDA_Tensor<float> CUDA_Tensor_float;
	typedef CUDA_Tensor<float*> CUDA_Tensor_float_ptr;
	typedef CUDA_Tensor<int> CUDA_Tensor_int;
	
	////////////////////////////////////////////////////////////
	///	TYPE SPECIFIC FUNCTIONS
	////////////////////////////////////////////////////////////
	void CUDA_Tensor_float_fill(CUDA_Tensor_float& t, float val);
	void CUDA_Tensor_float_fill(CUDA_Tensor_float& t, float mean, float dev);
	void CUDA_Tensor_float_scale(CUDA_Tensor_float& t, float factor);
	void CUDA_Tensor_float_diff(CUDA_Tensor_float& t1, CUDA_Tensor_float& t2, CUDA_Tensor_float& tout);
	void CUDA_Tensor_float_sum(CUDA_Tensor_float& t, CUDA_Tensor_float& tout);
	void CUDA_Tensor_float_copy(CUDA_Tensor_float& t, CUDA_Tensor_float& tout);

} /* namespace lg */

#endif /* end of include guard: TENSORCUDA_HPP */
