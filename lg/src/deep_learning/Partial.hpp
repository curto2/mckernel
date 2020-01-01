// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

#ifndef PARTIAL_HPP
#define PARTIAL_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include "Operation.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	class Partial : public Operation
	{
		public:
			Partial(const int size, const double connectivity);
			Partial(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void save(lg::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			void reset_deltas(const double momentum);
			void pruning(float alpha);
			float pruned_percent();
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(const int size, const double connectivity);
			
			#ifdef CUDA_BACKEND
			CUDA_Tensor_float _weights;
			CUDA_Tensor_float _bias;
			CUDA_Tensor_float _deltas;
			#else
			Tensor_float _weights;
			Tensor_float _bias;
			Tensor_float _deltas;
			#endif
			std::vector< std::vector< int > > _forward_map;
			int _input_size;
			double _connectivity;
	};

} /* namespace lg */

#endif /* end of include guard: PARTIAL_HPP */

