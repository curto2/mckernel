// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef ADDITION_HPP
#define ADDITION_HPP

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
	class Addition : public Operation
	{
		public:
			Addition();
			Addition(lg::IOData& data);
			void save(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();

			static std::shared_ptr<Operation> make();
		
		private:
			int _width, _height, _depth;
	};

} /* namespace lg */

#endif /* end of include guard: ADDITION_HPP */

