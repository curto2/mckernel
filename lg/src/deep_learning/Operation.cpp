// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

///////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Operation.hpp"
#include "Variable.hpp"
#include "Linear.hpp"
#include "Linear_Freeze.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "Recurrent.hpp"
#include "Partial.hpp"
#include "Dropout.hpp"
#include "Convolution.hpp"
#include "Normalization.hpp"
#include "Addition.hpp"
#include "Concatenate.hpp"
#include "Maxpooling.hpp"
#include "Averagepooling.hpp"
#include "Selu.hpp"
#include "Autoencoder.hpp"
#include "ResidualBlock.hpp"
#include "CapsulesDense.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	
	////////////////////////////////////////////////////////////
	Operation::~Operation() {}

	////////////////////////////////////////////////////////////
	void Operation::initialize(std::vector<Operation*> &inputs) {}
	
	////////////////////////////////////////////////////////////
	void Operation::save(lg::IOData& data) {}
	
	////////////////////////////////////////////////////////////
	void Operation::run(std::vector<Operation*>& inputs, const bool training) {}
	
	////////////////////////////////////////////////////////////
	void Operation::backprop(std::vector<Operation*>& input_errors) {}
	
	////////////////////////////////////////////////////////////
	void Operation::accumulate_deltas(std::vector<Operation*>& inputs) {}
	
	////////////////////////////////////////////////////////////
	void Operation::update_parameters(const float learningrate) {}
	
	////////////////////////////////////////////////////////////
	void Operation::print() {}
	
	////////////////////////////////////////////////////////////
	void Operation::reset_errors()
	{
		#ifdef CUDA_BACKEND
		if (_errors.size() > 0) CUDA_Tensor_float_fill(_errors, 0);
		#else
		for (int c = 0; c < _errors.size(); c++)
			_errors[c] = 0;
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Operation::reset_deltas(const double momentum)
	{
		/*
		#ifdef CUDA_BACKEND
		if (_deltas.size() > 0) CUDA_Tensor_float_scale(_deltas, momentum);
		#else
		for (int c = 0; c < _deltas.size(); c++)
			_deltas[c] *= momentum;
		#endif
		*/
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Operation::get_type() const { return Operation::Unknown; }
	
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Operation::loadFromFile(lg::IOData& data)
	{
		lg::IOData* op_type = data.findNode("operation_type");
		ensure(op_type != NULL);
		lg::IOData* op = data.findNode("operation_data");
		ensure(op != NULL);
		int operation_type;
		op_type->get(operation_type);
		switch (operation_type) {
			case Unknown: return std::shared_ptr<Operation>(NULL);
			case Variable: return std::shared_ptr<Operation>(new lg::Variable(*op));
			case Linear: return std::shared_ptr<Operation>(new lg::Linear(*op));
			case Linear_Freeze: return std::shared_ptr<Operation>(new lg::Linear_Freeze(*op));
			case Sigmoid: return std::shared_ptr<Operation>(new lg::Sigmoid(*op));
			case Relu: return std::shared_ptr<Operation>(new lg::Relu(*op));
			case Tanh: return std::shared_ptr<Operation>(new lg::Tanh(*op));
			case Softmax: return std::shared_ptr<Operation>(new lg::Softmax(*op));
			case Recurrent: return std::shared_ptr<Operation>(new lg::Recurrent(*op));
			case Partial: return std::shared_ptr<Operation>(new lg::Partial(*op));
			case Dropout: return std::shared_ptr<Operation>(new lg::Dropout(*op));
			case Convolution: return std::shared_ptr<Operation>(new lg::Convolution(*op));
			case Normalization: return std::shared_ptr<Operation>(new lg::Normalization(*op));
			case Addition: return std::shared_ptr<Operation>(new lg::Addition(*op));
			case Concatenate: return std::shared_ptr<Operation>(new lg::Concatenate(*op));
			case Maxpooling: return std::shared_ptr<Operation>(new lg::Maxpooling(*op));
			case Averagepooling: return std::shared_ptr<Operation>(new lg::Averagepooling(*op));
			case Selu: return std::shared_ptr<Operation>(new lg::Selu(*op));
			case Autoencoder: return std::shared_ptr<Operation>(new lg::Autoencoder(*op));
			case ResidualBlock: return std::shared_ptr<Operation>(new lg::ResidualBlock(*op));
			default: return std::shared_ptr<Operation>(NULL);
		}
	}
	
	////////////////////////////////////////////////////////////
	void Operation::saveToFile(std::shared_ptr<Operation>& operation, lg::IOData& data)
	{
		data.pushNode("operation_type", (int)operation.get()->get_type());
		data.pushNode("operation_data");
		lg::IOData* operation_data = data.findNode("operation_data");
		operation.get()->save(*operation_data);		
	}

} /* namespace lg */
