/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk						    */

// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza

#include "../../src/deep_learning/Neural_Network.hpp"
#include "../../src/deep_learning/SGD_Optimizer.hpp"
#include "../../src/datasets/MNIST_loader.hpp"
#include "../../../sdd+/hpp/Factory_McKernel.hpp"

int main(int argc, const char *argv[])
{    
	
	//Load training set
  	lg::Tensor_float trainingset, training_targets, testingset, testing_targets;
  	lg::MNIST_Load_Binary(
		"../../../data/mnist/train-images.idx3-ubyte",
		"../../../data/mnist/t10k-images.idx3-ubyte",
		"../../../data/mnist/train-labels.idx1-ubyte",
		"../../../data/mnist/t10k-labels.idx1-ubyte",
		trainingset, training_targets, testingset, testing_targets);


	printf("# Samples Test: %d\n", testingset.height());
	printf("Dimension Features: %d\n", testingset.width());
	printf("# Samples Train: %d\n", trainingset.height());
	printf("Dimension Features: %d\n", trainingset.width());

	//Kernel Expansions
	int expansions = 3;

	//RBF
	unsigned long t = 40;
	float sigma = 1.0;
	//float sigma = 10.0;

    	//Seed random distributions    	
	//random_device rd;
    	//unsigned long seed = (unsigned long)rd();
	//McKernel
	//printf("Seed: %lu...\n",seed);
	unsigned long seed = 1398239763;

	//McKernel
	printf("McKernel...\n");

	//SGD Optimizer
	const int batch_size = 10;
	lg::SGD_Optimizer sgd(batch_size, 0.001, 0.4, lg::Cost::CrossEntropy);

	//Train
	double error = 0;
	const int nepochs = 20;
	const int samples = trainingset.height();
	const int tsamples = testingset.height();
	const int cycles = nepochs * samples;
	
	const int restarts = 1;
	int best = 6000;
	for (int z = 0; z < restarts; z++) {
		
		//Reset learning rate
		float baselr = 0.001;
		sgd.setLearningrate(baselr);
		
		lg::Tensor_float input;
		lg::Tensor_float tinput;
		input.copy(trainingset.ptr(0, 0));
		tinput.copy(testingset.ptr(0, 0));

		//Initialize variables
		unsigned long nv = input.height();
		unsigned long dn = input.width();
		unsigned long D = expansions * dn;
		
		unsigned long tnv = tinput.height();
		unsigned long tdn = tinput.width();
		unsigned long tD = expansions * tdn;

		//Initialize McKernel
		McKernel* mckernel = FactoryMcKernel::createMcKernel(FactoryMcKernel::MRBF, input, nv, dn, D, seed, sigma, t);
		McKernel* tmckernel = FactoryMcKernel::createMcKernel(FactoryMcKernel::MRBF, tinput, tnv, tdn, tD, seed, sigma, t);	
		
		//Initialize variables
		lg::Tensor_float trainingset_McKernel(2 * mckernel->M_dn_D, input.height());
		lg::Tensor_float testingset_McKernel(2 * tmckernel->M_dn_D, tinput.height());
		trainingset_McKernel.fill(0);
		testingset_McKernel.fill(0);

		//Linear Classifier (1-layer Neural Network)
		lg::Neural_Network nn;
		nn.push("INPUT",		"",				lg::Variable::make(2 * tmckernel->M_dn_D));
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
			
			//Batch McKernel
			for (int c = 0; c < input.height(); c++)
				for (int k = 0; k < input.width(); k++)
					mckernel->M_data[c * input.width() + k] = input[c * input.width() + k];

			//McFeatures
			mckernel->McFeatures();

			//McEvaluate
			//data_features = mckernel->McEvaluate();
			mckernel->McEvaluate();

			//Tensor Float
			for (int c = 0; c < trainingset_McKernel.height(); c++)
				for (int k = 0; k < trainingset_McKernel.width(); k++)
					trainingset_McKernel[c * trainingset_McKernel.width() + k] = mckernel->M_features[c * trainingset_McKernel.width() + k];
		             
			error += nn.optimize(trainingset_McKernel, training_targets.ptr(0, random_sample_id), &sgd);

			if (c % 10000 == 0 && c != 0) {
				printf("Cycle: %d Error: %f Learning rate: %f\n", c, error / 10000.f, sgd.getLearningrate());
				error = 0;
			}


			if (c % samples == 0 && c != 0) {
				//Test
				printf("Epoch: %d Testing...\n", c / samples);
			
				int errors = 0;
		
				for (int c = 0; c < tsamples; c++) {
				
					tinput.copy(testingset.ptr(0, c));

					//Batch McKernel
					for (int c = 0; c < tinput.height(); c++)
						for (int k = 0; k < tinput.width(); k++)
							tmckernel->M_data[c * tinput.width() + k] = tinput[c * tinput.width() + k];

					//McFeatures
					tmckernel->McFeatures();

					//McEvaluate
					tmckernel->McEvaluate();

					//Tensor Float
					for (int c = 0; c < testingset_McKernel.height(); c++)
						for (int k = 0; k < testingset_McKernel.width(); k++)
							testingset_McKernel[c * testingset_McKernel.width() + k] = tmckernel->M_features[c * testingset_McKernel.width() + k];
		     
					nn.run(testingset_McKernel, false);
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
