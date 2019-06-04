// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

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
	class Softmax : public Operation
	{
		public:
			Softmax(double input_scale = 1.f);
			Softmax(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void save(lg::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(double input_scale = 1.f);

			float _input_scale;

		private:
			float _epsilon;
	};

} /* namespace lg */

#endif /* end of include guard: SOFTMAX_HPP */

