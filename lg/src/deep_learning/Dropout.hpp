// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef DROPOUT_HPP
#define DROPOUT_HPP

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
	class Dropout : public Operation
	{
		public:
			Dropout(const double drop_probability);
			Dropout(lg::IOData& data);
			void save(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(const double drop_probability);

			float _drop_probability;
		private:
			#ifdef CUDA_BACKEND
			lg::cudnn::Dropout _cuda_dropout;
			lg::CUDA_Tensor_float _state_buffer, _reserve_space_buffer;
			#endif
			int _width, _height, _depth;
	};

} /* namespace lg */

#endif /* end of include guard: DROPOUT_HPP */

