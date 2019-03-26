// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Cost.hpp"
#include <math.h>
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	Cost::Cost()
	{
		_type = Cost::SquaredError;
	}

	////////////////////////////////////////////////////////////
	Cost::Cost(const CostType type)
	{
		_type = type;
	}

	#ifdef CUDA_BACKEND
	////////////////////////////////////////////////////////////
	float Cost::getErrorcuda(CUDA_Tensor_float& prediction, CUDA_Tensor_float& target)
	{
		if (_gpu_errors.size() == 0) {
			_gpu_errors.setshape(target.size());
			_host_errors.setshape(target.size());
		}
		CUDA_Tensor_float_diff(target, prediction, _gpu_errors);
		_gpu_errors.copyToHost(_host_errors.pointer(), _host_errors.size());
		float error = 0;
		for (int c = 0; c < _host_errors.size(); c++)
			error += fabs(_host_errors[c]);
		return error;
		//TODO
	}

	////////////////////////////////////////////////////////////
	void Cost::getDeltacuda(CUDA_Tensor_float& prediction, CUDA_Tensor_float& target, CUDA_Tensor_float& errors)
	{
		switch (_type) {
			case Cost::SquaredError:
				CUDA_Tensor_float_diff(target, prediction, errors);
				break;
			
			case Cost::CrossEntropy:
				cuda::cost_crossentropy(prediction.pointer(), target.pointer(), errors.pointer(), errors.size());	
				break;
		}
	}

	#else
	
	////////////////////////////////////////////////////////////
	float Cost::getError(Tensor_float& prediction, Tensor_float& target)
	{
		float error = 0;
		switch (_type) {
			case Cost::SquaredError:
				for (int c = 0; c < (int)prediction.size(); c++)
					error += pow(target[c] - prediction[c], 2) / 2.f;
				break;
			
			case Cost::CrossEntropy:
				float epsilon = 1e-04;
				for (int c = 0; c < (int)prediction.size(); c++)
					error += target[c] * log(prediction[c] + epsilon) + (1 - target[c]) * log(1 - prediction[c] + epsilon);
				error = -error;
				break;
		}
		
		return error;
	}

	////////////////////////////////////////////////////////////
	void Cost::getDelta(Tensor_float& prediction, Tensor_float& target, Tensor_float& errors)
	{
		switch (_type) {
			case Cost::SquaredError:
				for (int c = 0; c < (int)errors.size(); c++)
					errors[c] = target[c] - prediction[c];
				break;
			
			case Cost::CrossEntropy:
				double denominator;
				double epsilon = 1e-4;
				for (int c = 0; c < (int)errors.size(); c++) {
					denominator = prediction[c] - prediction[c]*prediction[c];
					if (denominator < epsilon) denominator = epsilon;
					errors[c] = (target[c] - prediction[c]) / denominator;
				}
				break;
		}
	}

	#endif

} //namespace ai
