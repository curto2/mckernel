// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef RECURRENT_HPP
#define RECURRENT_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include "Operation.hpp"
#include "Linear.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	class Recurrent : public Operation
	{
		public:
			Recurrent(const int size, int btt_steps = 3);
			Recurrent(lg::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void initialize(const int input_size);
			void save(lg::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			const Operation::Type get_type() const;
			void reset_hidden_state();
			void print();
			
			static void gradient_check();
			static std::shared_ptr<Operation> make(const int size, int btt_steps = 3);
			
			//Weights and bias
			lg::Linear _x;
			lg::Linear _rec;
			lg::Linear _out;
		
			#ifdef CUDA_BACKEND
			
			void run(const CUDA_Tensor_float input, bool accumulate);
			void backprop(CUDA_Tensor_float out_errors);
			void accumulate_deltas(const CUDA_Tensor_float input);
			
			//Memory parameters for unrolling
			CUDA_Tensor_float _mem_outputs;
			CUDA_Tensor_float _mem_tmp_errors;
			CUDA_Tensor_float _mem_inputs;
			
			#else
		
			void run(const Tensor_float input, bool accumulate);
			void backprop(Tensor_float out_errors);
			void accumulate_deltas(const Tensor_float input);
			
			//Memory parameters for unrolling
			Tensor_float _mem_outputs;
			Tensor_float _mem_tmp_errors;
			Tensor_float _mem_inputs;
			
			#endif

			int _btt_steps;
			int _btt_pos;
			int _input_size;
	};

} /* namespace lg */

#endif /* end of include guard: RECURRENT_HPP */

