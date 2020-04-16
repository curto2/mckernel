// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef SIGMOID_HPP
#define SIGMOID_HPP

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
	class Sigmoid : public Operation
	{
		public:
			Sigmoid();
			Sigmoid(lg::IOData& data);
			void save(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make();
			
			////////////////////////////////////////////////////////////
			///	RAW OPERATIONS
			////////////////////////////////////////////////////////////
			#ifndef CUDA_BACKEND
			static void forward(const Tensor_float input, Tensor_float output);
			static void backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors);
			#else
			lg::cudnn::Activation _cudnnactivation;
			#endif
		
		private:
			int _width, _height, _depth;
	};

} /* namespace lg */

#endif /* end of include guard: SIGMOID_HPP */

