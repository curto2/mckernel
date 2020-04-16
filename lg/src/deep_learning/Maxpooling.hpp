// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef MAXPOOLING_HPP
#define MAXPOOLING_HPP

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
	class Maxpooling : public Operation
	{
		public:
			Maxpooling(const int filter_size, const int stride);
			Maxpooling(lg::IOData& data);
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
			std::vector<int> _maxin;
			#endif
	};

} /* namespace lg */

#endif /* end of include guard: MAXPOOLING_HPP */

