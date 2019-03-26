/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk						    */

// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza


#include "../../src/AI/deeplearning/Neural_Network.hpp"
#include "../../src/AI/deeplearning/SGD_Optimizer.hpp"
#include "../../src/AI/datasets/MNIST_loader.hpp"
#include "../../../sdd+/hpp/Factory_McKernel.hpp"

int main(int argc, const char *argv[])
{    
	
	//Load training set
  	ai::Tensor_float trainingset, training_targets, testingset, testing_targets;
  	ai::MNIST_Load_Binary(
		"../../../data/mnist/train-images-idx3-ubyte",
		"../../../data/mnist/t10k-images-idx3-ubyte",
		"../../../data/mnist/train-labels-idx1-ubyte",
		"../../../data/mnist/t10k-labels-idx1-ubyte",
		trainingset, training_targets, testingset, testing_targets);

	printf("%d\n", testingset.height());
	printf("%d\n", testingset.width());
	printf("%d\n", trainingset.height());
	printf("%d\n", trainingset.width());


	//McKernel
	printf("McKernel...\n");

	//SGD Optimizer
	const int batch_size = 10;
	ai::SGD_Optimizer sgd(batch_size, 0.01, 0.5, ai::Cost::CrossEntropy);

	//Train
	double error = 0;
        const int nepochs = 20;
        const int samples = trainingset.height();
        const int cicles = nepochs * samples;
	
	const int restarts = 1;
	int best = 9000;
	for (int z = 0; z < restarts; z++) {
		
		//Reset learning rate
		float baselr = 0.01;
		sgd.setLearningrate(baselr);
		
		ai::Tensor_float input;
		ai::Tensor_float tinput;

		//Logistic Regression
		ai::Neural_Network nn;
		nn.push("INPUT",		"",				ai::Variable::make(28,28));
		nn.push("LINEAR",		"INPUT",			ai::Linear::make(10));
		nn.push("OUTPUT",		"LINEAR",			ai::Sigmoid::make());

		//Print Network 
		nn.printstack();
	
		for (int c = 0; c <= cicles; c++) {

		//Update learning rate
		sgd.setLearningrate(sgd.getLearningrate() - baselr / (double)cicles);

		//Optimize neural network with random sample
		int random_sample_id = rand() % samples;
		input.copy(trainingset.ptr(0, random_sample_id));
                     
			error += nn.optimize(input, training_targets.ptr(0, random_sample_id), &sgd);

			if (c % 10000 == 0 && c != 0) {
				printf("Cicle: %d Error: %f Learning rate: %f\n", c, error / 10000.f, sgd.getLearningrate());
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
					printf("Network saved!\n");
				}
			}
		}
	}
	
	printf("Complete Learning\n");
	return 0;
}
