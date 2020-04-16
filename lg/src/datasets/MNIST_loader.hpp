// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef LMNIST_HPP
#define LMNIST_HPP

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
	void MNIST_Load(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels);
	
	void MNIST_Load_Binary(const std::string train_images_path, const std::string test_images_path,
		const std::string train_labels_path, const std::string test_labels_path, 
		Tensor_float& trainingset, Tensor_float& training_labels, Tensor_float& testingset, Tensor_float& testing_labels);

} /* namespace lg */

#endif /* end of include guard: LMNIST_HPP */

