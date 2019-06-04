// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "CUDA_Tensor.hpp"
#include <stdio.h>
#include "Macros.hpp"
#include "ensure.hpp"
#include <memory>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	///	ERROR HANDLING
	////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////
	void HandleError(cudaError_t err, const char* file, int line)
	{
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
			exit(EXIT_FAILURE);
		}
	}
	
	#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))
	
	////////////////////////////////////////////////////////////
	///	UTIL
	////////////////////////////////////////////////////////////

	//Get nearest lower power of two
	unsigned int low_pow2 (unsigned int x)
	{
		x = x | (x >> 1);
		x = x | (x >> 2);
		x = x | (x >> 4);
		x = x | (x >> 8);
		x = x | (x >> 16);
		return x - (x >> 1);
	}

	//Get nearest higher power of two
	unsigned long high_pow2(unsigned long v)
	{
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}
	
	////////////////////////////////////////////////////////////
	///	KERNELS
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	__global__ void knl_tensor_fill(float* t, float val, int size)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		while (tid < size) {
			t[tid] = val;
			tid += blockDim.x * gridDim.x;
		}
	}
	
	////////////////////////////////////////////////////////////
	__global__ void knl_tensor_fill_random(float* t, float mean, float dev, unsigned int seed, int size)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tseed = seed * (tid + 1);
		while (tid < size) {
			tseed ^= tseed << 13;
			tseed ^= tseed >> 17;
			tseed ^= tseed << 5;
			t[tid] = mean - dev + ((float)tseed / UINT_MAX) * dev * 2.f;
			tid += blockDim.x * gridDim.x;
		}
	}

	////////////////////////////////////////////////////////////
	__global__ void knl_tensor_scale(float* t, float factor, int size)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		while (tid < size) {
			t[tid] *= factor;
			tid += blockDim.x * gridDim.x;
		}
	}

	////////////////////////////////////////////////////////////
	__global__ void knl_tensor_diff(float* t1, float* t2, float* tout, int size)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		while (tid < size) {
			tout[tid] = t1[tid] - t2[tid];
			tid += blockDim.x * gridDim.x;
		}
	}
	
	////////////////////////////////////////////////////////////
	__global__ void knl_tensor_add(float* t1, float* t2, int size)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		while (tid < size) {
			t1[tid] += t2[tid];
			tid += blockDim.x * gridDim.x;
		}
	}
	
	////////////////////////////////////////////////////////////
	__global__ void knl_tensor_copy(float* t1, float* t2, int size)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		while (tid < size) {
			t1[tid] = t2[tid];
			tid += blockDim.x * gridDim.x;
		}
	}

	////////////////////////////////////////////////////////////
	///	TENSOR GPU
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T>::CUDA_Tensor()
	{
		_data = NULL;
		_size = 0;
		_depth = _height = _width = 0;
		_owner = false;
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T>::CUDA_Tensor(const CUDA_Tensor<T>& t)
	{
		point(t);
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T>::CUDA_Tensor(int width)
	{
		_width = width;
		_height = 1;
		_depth = 1;
		_size = _width * _depth * _height;
		_owner = true;
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T) ));
	}

	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T>::CUDA_Tensor(int width, int height)
	{
		_width = width;
		_height = height;
		_depth = 1;
		_size = _width * _depth * _height;
		_owner = true;
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
	}	

	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T>::CUDA_Tensor(int width, int height, int depth)
	{
		_width = width;
		_height = height;
		_depth = depth;
		_size = _width * _depth * _height;
		_owner = true;
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
	}

	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T>::~CUDA_Tensor()
	{
		clear();
	}
	
	////////////////////////////////////////////////////////////
	template <typename T>
	void CUDA_Tensor<T>::load(lg::IOData& data, std::string dataname)
	{
		clear();
		IOData* node_width = data.findNode(dataname + "_width");	
		IOData* node_height = data.findNode(dataname + "_height");	
		IOData* node_depth = data.findNode(dataname + "_depth");
		IOData* node_data = data.findNode(dataname + "_data");
		ensure(node_width != NULL);
		ensure(node_height != NULL);
		ensure(node_depth != NULL);
		ensure(node_data != NULL);
		node_width->get(_width);
		node_height->get(_height);
		node_depth->get(_depth);
		_size = _width * _height * _depth;
		std::unique_ptr<T> tmp = std::unique_ptr<T>(new T[_size]);
		node_data->get(reinterpret_cast<char*>(&tmp.get()[0]));
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
		copyToDevice(&tmp.get()[0], _size);
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::load(std::ifstream& file)
	{
		clear();
		file.read(reinterpret_cast<char*>(&_size), sizeof(_size));
		file.read(reinterpret_cast<char*>(&_width), sizeof(_width));
		file.read(reinterpret_cast<char*>(&_height), sizeof(_height));
		file.read(reinterpret_cast<char*>(&_depth), sizeof(_depth));
		_owner = true;
		std::unique_ptr<T> tmp = std::unique_ptr<T>(new T[_size]);
		file.read(reinterpret_cast<char*>(&tmp.get()[0]), sizeof(T) * _size);
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
		copyToDevice(&tmp.get()[0], _size);
	}
	
	////////////////////////////////////////////////////////////
	template <typename T>
	void CUDA_Tensor<T>::save(lg::IOData& data, std::string dataname)
	{
		std::unique_ptr<T> tmp_safe = std::unique_ptr<T>(new T[_size]);
		T* tmp = tmp_safe.get();
		copyToHost(&tmp[0], _size);
		data.pushNode(dataname + "_width", _width);	
		data.pushNode(dataname + "_height", _height);	
		data.pushNode(dataname + "_depth", _depth);
		data.pushNode(dataname + "_data", reinterpret_cast<char*>(&tmp[0]), sizeof(T) * _size);
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::save(std::ofstream& file)
	{
		std::unique_ptr<T> tmp_safe = std::unique_ptr<T>(new T[_size]);
		T* tmp = tmp_safe.get();
		copyToHost(&tmp[0], _size);
		file.write(reinterpret_cast<char*>(&_size), sizeof(_size));
		file.write(reinterpret_cast<char*>(&_width), sizeof(_width));
		file.write(reinterpret_cast<char*>(&_height), sizeof(_height));
		file.write(reinterpret_cast<char*>(&_depth), sizeof(_depth));
		file.write(reinterpret_cast<char*>(&tmp[0]), sizeof(T) * _size);
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::setshape(const int width)
	{
        ensure(width > 0);
        clear();
        _width = width;
        _height = 1;
        _depth = 1;
        _size = _width * _height * _depth;
		_owner = true;
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::setshape(const int width, const int height)
	{
        ensure(width > 0 && height > 0);
        clear();
        _width = width;
        _height = height;
        _depth = 1;
        _size = _width * _height * _depth;
		_owner = true;
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::setshape(const int width, const int height, const int depth)
	{
        ensure(width > 0 && height > 0 && depth > 0);
        clear();
        _width = width;
        _height = height;
        _depth = depth;
        _size = _width * _height * _depth;
		_owner = true;
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
	}
    
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::setshape(Tensor<T>& host_tensor)
	{
        clear();
        _width = host_tensor.width();
        _height = host_tensor.height();
        _depth = host_tensor.depth();
        _size = _width * _height * _depth;
		_owner = true;
		HANDLE_ERROR( cudaMalloc( &_data, _size * sizeof(T)) );
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::point(const CUDA_Tensor& t)
	{
        clear();
        _data = t._data;
        _size = t._width * t._height * t._depth;
        _width = t._width;
        _height = t._height;
        _depth = t._depth;
        _owner = false;
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::point(const CUDA_Tensor& t, const unsigned int offset_d)
	{
        clear();
        _data = &t._data[offset_d * t._width * t._height];
        _size = t._width * t._height;
        _width = t._width;
        _height = t._height;
        _depth = 1;
        _owner = false;
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::point(const CUDA_Tensor& t, const unsigned int offset_d, const unsigned int offset_y)
	{
        clear();
        _data = &t._data[offset_d * t._width * t._height + offset_y * t._width];
        _size = t._width;
        _width = t._width;
        _height = 1;
        _depth = 1;
        _owner = false;
	}

	//////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::clear()
	{
		if (_data != NULL && _size != 0 && _owner == true)
			HANDLE_ERROR( cudaFree(_data) );
		_data = NULL;
		_size = 0;
		_width = 0;
		_height = 0;
		_depth = 0;
		_owner = false;
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::fill(T val)
	{
		std::unique_ptr<T> tmp_safe = std::unique_ptr<T>(new T[_size]);
		T* temp = tmp_safe.get();
		for (int c = 0; c < _size; c++) temp[c] = val;
		copyToDevice(temp, _size);
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::fill(float mean, float dev)
	{
		std::unique_ptr<T> tmp_safe = std::unique_ptr<T>(new T[_size]);
		T* temp = tmp_safe.get();
		for (int c = 0; c < _size; c++)
			temp[c] = (T)( mean - dev + ((double)rand() / RAND_MAX) * dev * 2.f);
		copyToDevice(temp, _size);
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::copyToHost(T *arr, int size) const
	{
		ensure(size <= _size);
		HANDLE_ERROR( cudaMemcpy( arr, _data, size * sizeof(T), cudaMemcpyDeviceToHost));
	}

	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::copyToDevice(const T *arr, int size)
	{
		ensure(size <= _size);
		HANDLE_ERROR( cudaMemcpy(_data, arr,  size * sizeof(T), cudaMemcpyHostToDevice));
	}
	

	////////////////////////////////////////////////////////////
	template < typename T >
	void CUDA_Tensor<T>::copy(const CUDA_Tensor<T>& tensor)
	{
		if (width() != tensor.width() || height() != tensor.height() || depth() != tensor.height())
			setshape((int)tensor.width(), (int)tensor.height(), (int)tensor.depth());
		HANDLE_ERROR( cudaMemcpy(_data, tensor.pointer(),  tensor.size() * sizeof(T), cudaMemcpyDeviceToDevice));
	}
	
	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T> CUDA_Tensor<T>::ptr(const int d)
	{
		CUDA_Tensor<T> t;
		t.point(*this, d);
		return t;
	}

	////////////////////////////////////////////////////////////
	template < typename T >
	CUDA_Tensor<T> CUDA_Tensor<T>::ptr(const int d, const int y)
	{
		CUDA_Tensor<T> t;
		t.point(*this, d, y);
		return t;
	}
	
	////////////////////////////////////////////////////////////
	///	TYPE SPECIFIC FUNCTIONS
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	void CUDA_Tensor_float_fill(CUDA_Tensor_float& t, float val)
	{
		ensure(t.pointer() != NULL && t.size() != 0);
		int _threads = min(low_pow2(t.size()), CUDA_MAX_THREADS);
		int _blocks = min(_threads / t.size() + 1, CUDA_MAX_CORES);
		knl_tensor_fill<<<_blocks, _threads>>>(t.pointer(), val, t.size());
	}
	
	////////////////////////////////////////////////////////////
	void CUDA_Tensor_float_fill(CUDA_Tensor_float& t, float mean, float dev)
	{
		ensure(t.pointer() != NULL && t.size() != 0);
		int _threads = min(low_pow2(t.size()), CUDA_MAX_THREADS);
		int _blocks = min(_threads / t.size() + 1, CUDA_MAX_CORES);
		knl_tensor_fill_random<<<_blocks, _threads>>>(t.pointer(), mean, dev, rand(), t.size());
	}

	////////////////////////////////////////////////////////////
	void CUDA_Tensor_float_scale(CUDA_Tensor_float& t, float factor)
	{
		ensure(t.pointer() != NULL && t.size() != 0);
		int _threads = min(low_pow2(t.size()), CUDA_MAX_THREADS);
		int _blocks = min(_threads / t.size() + 1, CUDA_MAX_CORES);
		knl_tensor_scale<<<_blocks, _threads>>>(t.pointer(), factor, t.size());
	}

	////////////////////////////////////////////////////////////
	void CUDA_Tensor_float_diff(CUDA_Tensor_float& t1, CUDA_Tensor_float& t2, CUDA_Tensor_float& tout)
	{
		ensure(tout.pointer() != NULL && t1.pointer() != NULL && t2.pointer() != NULL &&
			(tout.size() == t1.size() && tout.size() == t2.size()));
		int _threads = min(low_pow2(tout.size()), CUDA_MAX_THREADS);
		int _blocks = min(_threads / tout.size() + 1, CUDA_MAX_CORES);
		knl_tensor_diff<<<_blocks, _threads>>>(t1.pointer(), t2.pointer(), tout.pointer(), tout.size());
	}

	////////////////////////////////////////////////////////////
	void CUDA_Tensor_float_sum(CUDA_Tensor_float& t, CUDA_Tensor_float& tout)
	{
		ensure(tout.pointer() != NULL && t.pointer() != NULL && tout.size() == t.size());
		int _threads = min(low_pow2(tout.size()), CUDA_MAX_THREADS);
		int _blocks = min(_threads / tout.size() + 1, CUDA_MAX_CORES);
		knl_tensor_add<<<_blocks, _threads>>>(tout.pointer(), t.pointer(), tout.size());
	}

	////////////////////////////////////////////////////////////
	void CUDA_Tensor_float_copy(CUDA_Tensor_float& t, CUDA_Tensor_float& tout)
	{
		ensure(tout.pointer() != NULL && t.pointer() != NULL && tout.size() == t.size());
		int _threads = min(low_pow2(tout.size()), CUDA_MAX_THREADS);
		int _blocks = min(_threads / tout.size() + 1, CUDA_MAX_CORES);
		knl_tensor_copy<<<_blocks, _threads>>>(tout.pointer(), t.pointer(), tout.size());
	}
	
	//Specialization
	template < > void CUDA_Tensor<float*>::fill(float mean, float dev) { }
	
	//Explicit instantiations
	template class CUDA_Tensor<float>;
	template class CUDA_Tensor<float*>;
	template class CUDA_Tensor<int>;
	
	////////////////////////////////////////////////////////////
	

} //namespace lg
