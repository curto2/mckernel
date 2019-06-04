// Original Source Code by Meroni (https://github.com/Flowx08/)
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
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	void CIFAR10_Load(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels);

} /* namespace lg */

#endif /* end of include guard: LCIFAR_HPP */

