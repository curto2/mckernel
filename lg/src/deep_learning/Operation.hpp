// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef OPERATION_HPP
#define OPERATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <fstream>
#include <vector>
#include <memory>
#include "../util/Tensor.hpp"
#include "../util/IOData.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "../util/CUDA_Tensor.hpp"
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	class Operation
	{
		public:

			////////////////////////////////////////////////////////////
			///	OPERATIONS TYPES AVAILABLE
			////////////////////////////////////////////////////////////
			enum Type
			{
				Unknown,
				Variable,
				Linear,
				Linear_Freeze,
				Sigmoid,
				Tanh,
				Relu,
				Softmax,
				Recurrent,
				Partial,
				Dropout,
				Convolution,
				Normalization,
				Addition,
				Concatenate,
				Maxpooling,
				Averagepooling,
				Selu,
				Autoencoder,
				ResidualBlock,
				CapsulesDense,
				Types_Count
			};

			virtual ~Operation();
			virtual void initialize(std::vector<Operation*> &inputs);
			virtual void run(std::vector<Operation*>& inputs, const bool training);
			virtual void backprop(std::vector<Operation*>& inputs);
			virtual void accumulate_deltas(std::vector<Operation*>& inputs);
			virtual void update_parameters(const float learningrate);
			virtual void print();
			virtual const Type get_type() const;
			virtual void reset_deltas(const double momentum);
			virtual void reset_errors();

			static std::shared_ptr<Operation> loadFromFile(lg::IOData& data);
			static void saveToFile(std::shared_ptr<Operation>& operation, lg::IOData& data);
			
			int _size;
			#ifdef CUDA_BACKEND
			CUDA_Tensor_float _outputs;
			CUDA_Tensor_float _errors;
			#else
			Tensor_float _outputs;
			Tensor_float _errors;
			#endif

		private:
			virtual void save(lg::IOData& data);
	};

} /* namespace lg */

#endif /* end of include guard: OPERATION_HPP */

