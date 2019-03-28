// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef LCIFAR_HPP
#define LCIFAR_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include "../util/Tensor.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	void CIFAR10_Load(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels);

} /* namespace ai */

#endif /* end of include guard: LCIFAR_HPP */

