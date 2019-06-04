// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include "Operation.hpp"
#include "Node_Network.hpp"
#include "Optimizer.hpp"
#include "../util/Macros.hpp"
#include <memory>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	class Neural_Network
	{
		public:
			Neural_Network();
			Neural_Network(std::string filepath);
			~Neural_Network();
			void save(std::string filepath);
			void load(std::string filepath);
			void push(std::string node_name, std::string inputs_names, std::shared_ptr<Operation> operation);
			void clear();
			Node_Network* get_byname(std::string node_name);
			#ifdef CUDA_BACKEND
			void run(CUDA_Tensor_float input, const bool training = false);
			float optimize(CUDA_Tensor_float input, CUDA_Tensor_float target, Optimizer* opt);
			bool test(CUDA_Tensor_float input, CUDA_Tensor_float target);
			CUDA_Tensor_float& get_output(std::string node_name);
			#else
			void run(Tensor_float input, const bool training = false);
			float optimize(Tensor_float input, Tensor_float target, Optimizer* opt);
			Tensor_float& get_output(std::string node_name);
			#endif
			std::vector<Node_Network>& getNodes();
			void printstack();
			
		private:
			void resetOperationsErrors();
			std::vector<std::string> splitString(std::string s, char delimiter);

			std::vector<Node_Network> _nodes;
	};

} /* namespace lg */

#endif /* end of include guard: NEURALNETWORK_HPP */

