// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "MNIST_loader.hpp"
#include "../util/Files.hpp"
#include "../visualization/Bitmap.hpp"
#include "CSV.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{

	void CIFAR10_Load(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels)
	{
		const int image_area = 32 * 32;
		const int image_channels = 3;
		const int sample_size = image_area * image_channels;
		const int target_size = 10;

		//Load labels from CSV files
		int trainingset_size = 0, testingset_size = 0;
		std::vector< std::vector< std::string > > CSV_training = lg::CSV_Load(folder_path + "/training_labels.csv");
		std::vector< std::vector< std::string > > CSV_testing = lg::CSV_Load(folder_path + "/testing_labels.csv");
		trainingset_size = CSV_training.size();
		testingset_size = CSV_testing.size();

		//Allocate tensors
		trainingset.setshape(sample_size, trainingset_size);
		training_labels.setshape(target_size, trainingset_size);
		training_labels.fill(0);
		testingset.setshape(sample_size, testingset_size);
		testing_labels.setshape(target_size, testingset_size);
		testing_labels.fill(0);
		
		//Log
		printf("Loading training set...\n");

		//Load and normalize all the training images and compute the labels
		for (int z = 0; z < trainingset_size; z++) {
			//Bitmap bm(folder_path + "/training/" + std::to_string(z) + ".png", Bitmap::RGB);
			Bitmap bm(folder_path + "/" + CSV_training[z][0].c_str(), Bitmap::RGB);
			for (int c = 0; c < image_channels; c++)
				for (int x = 0; x < image_area; x++)
					trainingset.at(z, c * image_area + x) = bm.getData()[x * image_channels + c] / 255.f;
			training_labels.at(z, std::stoi(CSV_training[z][1])) = 1.f;
			if (z % (trainingset_size / 10) == 0) printf("Progress: %f\n", (double)z / (double)trainingset_size);
		}
		
		//Log
		printf("Loading testing set...\n");
		
		//Load and normalize all the testing images and compute the labels
		for (int z = 0; z < testingset_size; z++) {
			//Bitmap bm(folder_path + "/testing/" + std::to_string(z) + ".png", Bitmap::RGB);
			Bitmap bm(folder_path + "/" + CSV_testing[z][0].c_str(), Bitmap::RGB);
			for (int c = 0; c < image_channels; c++)
				for (int x = 0; x < image_area; x++)
					testingset.at(z, c * image_area + x) = bm.getData()[x * image_channels + c] / 255.f;
			testing_labels.at(z, std::stoi(CSV_testing[z][1].c_str())) = 1.f;
			if (z % (testingset_size / 10) == 0) printf("Progress: %f\n", (double)z / (double)testingset_size);
		}
	}

} /* namespace lg */
