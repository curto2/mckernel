/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time.		    

   Authors: Curtó and Zarza.
   c@decurto.tw z@dezarza.tw 						    */

// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.

#include "../../src/deep_learning/Neural_Network.hpp"
#include "../../src/deep_learning/SGD_Optimizer.hpp"
#include "../../src/datasets/MNIST_loader.hpp"

int main(int argc, const char *argv[])
{    
	
	//Load training set
  	lg::Tensor_float trainingset, training_targets, testingset, testing_targets;
  	lg::MNIST_Load_Binary(
		"../../../data/mnist/train-images-idx3-ubyte",
		"../../../data/mnist/t10k-images-idx3-ubyte",
		"../../../data/mnist/train-labels-idx1-ubyte",
		"../../../data/mnist/t10k-labels-idx1-ubyte",
		trainingset, training_targets, testingset, testing_targets);

	printf("# Samples Test: %d.\n", testingset.height());
	printf("Dimension Features: %d.\n", testingset.width());
	printf("# Samples Train: %d.\n", trainingset.height());
	printf("Dimension Features: %d.\n\n", trainingset.width());

	//SGD Optimizer
	const int batch_size = 10;
	lg::SGD_Optimizer sgd(batch_size, 0.01, 0.5, lg::Cost::CrossEntropy);

	//Train
	double error = 0;
	const int nepochs = 20;
	const int samples = trainingset.height();
	const int cycles = nepochs * samples;
	
	const int restarts = 1;
	int best = 9000;
	for (int z = 0; z < restarts; z++) {
		
		//Reset learning rate
		float baselr = 0.01;
		sgd.setLearningrate(baselr);
		
		lg::Tensor_float input;
		lg::Tensor_float tinput;

		//Logistic Regression
		lg::Neural_Network nn;
		nn.push("INPUT",		"",				lg::Variable::make(28,28));
		nn.push("LINEAR",		"INPUT",			lg::Linear::make(10));
		nn.push("OUTPUT",		"LINEAR",			lg::Sigmoid::make());

		//Print Network 
		nn.printstack();
	
		for (int c = 0; c <= cycles; c++) {

		//Update learning rate
		sgd.setLearningrate(sgd.getLearningrate() - baselr / (double)cycles);

		//Optimize neural network with random sample
		int random_sample_id = rand() % samples;
		input.copy(trainingset.ptr(0, random_sample_id));
                     
			error += nn.optimize(input, training_targets.ptr(0, random_sample_id), &sgd);

			if (c % 10000 == 0 && c != 0) {
				printf("Cycle: %d Error: %f Learning rate: %f\n", c, error / 10000.f, sgd.getLearningrate());
				error = 0;
			}


			if (c % samples == 0 && c != 0) {

				//Test
				printf("Epoch: %d Testing...\n", c / samples);
				
				int errors = 0;
                                int tsamples = (int)testingset.height();
			
				for (int c = 0; c < tsamples; c++) {
					
					tinput.copy(testingset.ptr(0, c));

					nn.run(tinput, false);
					int maxid;
					nn.get_output("OUTPUT").max(NULL, &maxid);
					int target_id;
					testing_targets.ptr(0, c).max(NULL, &target_id);
					if (maxid != target_id) errors++;
				}
				printf("Samples: %d Errors: %d Accuracy: %f\n", tsamples, errors, 1.f - (double)errors / (double)tsamples);

				if (errors < best) {
					best = errors;
					nn.save("MNIST.nn");
					printf("Saved network!\n");
				}
			}
		}
	}
	
	printf("Complete Learning.\n");
	return 0;
}
