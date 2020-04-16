// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef CONCATENATE_HPP
#define CONCATENATE_HPP

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
	class Concatenate : public Operation
	{
		public:
			Concatenate();
			Concatenate(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void save(lg::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make();
			
		private:
			int _width, _height, _depth;
			#ifdef CUDA_BACKEND
			CUDA_Tensor_float_ptr _inputs_pointers;
			CUDA_Tensor_float_ptr _outerrors_pointers;
			CUDA_Tensor_int _pointers_sizes;
			#endif
	};

} /* namespace lg */

#endif /* end of include guard: CONCATENATE_HPP */

