// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

#ifndef AVERAGEPOOLING_HPP
#define AVERAGEPOOLING_HPP

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
	class Averagepooling : public Operation
	{
		public:
			Averagepooling(const int filter_size, const int stride);
			Averagepooling(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void save(lg::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void print();
			const Operation::Type get_type() const;
			static std::shared_ptr<Operation> make(const int filter_size, const int stride);
		
			int _input_width;
			int _input_height;
			int _input_count;
			int _filter_size;
			int _stride;
			int _output_width;
			int _output_height;
			int _output_size;
			int _input_size;
			
			#ifdef CUDA_BACKEND
			lg::cudnn::Pooling _cuda_pooling;
			#else
			Tensor_float _average_in;
			#endif
	};

} /* namespace lg */

#endif /* end of include guard: AVERAGEPOOLING_HPP */

