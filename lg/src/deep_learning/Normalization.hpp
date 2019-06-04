// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include "Operation.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	class Normalization : public Operation
	{
		public:
			Normalization(float momentum = 0.1);
			Normalization(lg::IOData& data);
			void save(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(float momentum = 0.1);

		private:
			int _width, _height, _depth;
			float _gamma;
			float _beta;
			float _epsilon;
			float _momentum;
			
			//forward informations
			double _mean;
			double _variance;
			#ifdef CUDA_BACKEND
			CUDA_Tensor_float _deviation;
			CUDA_Tensor_float _normalized;
			CUDA_Tensor_float _params;
			#else
			Tensor_float _deviation;
			Tensor_float _normalized;
			#endif

			//Backward informations
			float _d_beta;
			float _d_gamma;
	};

} /* namespace lg */

#endif /* end of include guard: NORMALIZATION_HPP */

