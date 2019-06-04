// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "MNIST_loader.hpp"
#include "MNIST_binary_loader.hpp"
#include "../util/Files.hpp"
#include "../visualization/Bitmap.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	void MNIST_Load(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels)
	{
		const int digits_count = 10;
		const int sample_size = 28 * 28;
		const int target_size = 10;
		
		//Get all file names and calculate size of training set and testing set
		int trainingset_size = 0, testingset_size = 0;
		std::vector< std::vector<std::string> > training_files(digits_count);
		for (int c = 0; c < digits_count; c++) {
			training_files[c] = lg::files::listdir(folder_path + "/training/" + std::to_string(c) + "/");
			trainingset_size += (int)training_files[c].size();
		}
		std::vector< std::vector<std::string> > testing_files(digits_count);
		for (int c = 0; c < digits_count; c++) {
			testing_files[c] = lg::files::listdir(folder_path + "/testing/" + std::to_string(c) + "/");
			testingset_size += (int)testing_files[c].size();
		}
		
		//Allocate tensors
		trainingset.setshape(sample_size, trainingset_size);
		training_labels.setshape(target_size, trainingset_size);
		training_labels.fill(0);
		testingset.setshape(sample_size, testingset_size);
		testing_labels.setshape(target_size, testingset_size);
		testing_labels.fill(0);

		//Log
		printf("Loading training set...\n");

		//Load and normalize all the training images and compute labels
		int offset = 0;
		for (int c = 0; c < digits_count; c++) {
			for (int k = 0; k < (int)training_files[c].size(); k++) {
				Bitmap bm(folder_path + "/training/" + std::to_string(c) + "/" + training_files[c][k], Bitmap::MONO);
				bm.convertToMono();
				for (int z = 0; z < sample_size; z++)
					trainingset.at(offset + k, z) = bm.getData()[z] / 255.f;
				training_labels.at(offset + k, c) = 1.f;
			}
			offset += (int)training_files[c].size();
			printf("Progress: %f\n", (double)offset / (double)trainingset_size);
		}
		
		//Log
		printf("Loading testing set...\n");
		
		//Load and normalize all testing images and compute labels
		offset = 0;
		for (int c = 0; c < digits_count; c++) {
			for (int k = 0; k < (int)testing_files[c].size(); k++) {
				Bitmap bm(folder_path + "/testing/" + std::to_string(c) + "/" + testing_files[c][k], Bitmap::MONO);
				bm.convertToMono();
				for (int z = 0; z < sample_size; z++)
					testingset.at(offset + k, z) = bm.getData()[z] / 255.f;
				testing_labels.at(offset + k, c) = 1.f;
			}
			offset += (int)testing_files[c].size();
			printf("Progress: %f\n", (double)offset / (double)testingset_size);
		}
	}
	
	void MNIST_Load_Binary(const std::string train_images_path, const std::string test_images_path,
		const std::string train_labels_path, const std::string test_labels_path, 
		Tensor_float& trainingset, Tensor_float& training_labels, Tensor_float& testingset, Tensor_float& testing_labels)
	{
		printf("MNIST dataset loading...\n");
		
		//Load all raw data
		MNIST_binary_loader mnist(train_images_path, test_images_path, train_labels_path, test_labels_path);
		
		//Allocate training set tensor and training labels tensor
		trainingset.setshape(28 * 28, (int)mnist.get_train_images().size());
		training_labels.setshape(10, (int)mnist.get_train_images().size());
		training_labels.fill(0);

		//Fill training set tensor
		for (int c = 0; c < (int)mnist.get_train_images().size(); c++)
			for (int k = 0; k < 28 * 28; k++)
				trainingset.at(c, k) = mnist.get_train_images()[c][k] / 255.f;
		
		//Fill training labels tensor
		for (int c = 0; c < (int)mnist.get_train_images().size(); c++)
			training_labels.at(c, (int)mnist.get_train_labels()[c]) = 1.f;
		
		//Allocate testing set tensor and testing labels tensor
		testingset.setshape(28 * 28, (int)mnist.get_test_images().size());
		testing_labels.setshape(10, (int)mnist.get_test_images().size());
		testing_labels.fill(0);
		
		//Fill testing set tensor
		for (int c = 0; c < (int)mnist.get_test_images().size(); c++)
			for (int k = 0; k < 28 * 28; k++)
				testingset.at(c, k) = mnist.get_test_images()[c][k] / 255.f;
		
		//Fill testing labels tensor
		for (int c = 0; c < (int)mnist.get_test_images().size(); c++)
			testing_labels.at(c, (int)mnist.get_test_labels()[c]) = 1.f;
	}

} /* namespace lg */
