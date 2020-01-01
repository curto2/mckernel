/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time.		    

   Authors: Curtó and Zarza.
   c@decurto.ch z@dezarza.ch 						    */
                                                                     
_|      _|            _|    _|                                          _|  
_|_|  _|_|    _|_|_|  _|  _|      _|_|    _|  _|_|  _|_|_|      _|_|    _|  
_|  _|  _|  _|        _|_|      _|_|_|_|  _|_|      _|    _|  _|_|_|_|  _|  
_|      _|  _|        _|  _|    _|        _|        _|    _|  _|        _|  
_|      _|    _|_|_|  _|    _|    _|_|_|  _|        _|    _|    _|_|_|  _|
         				     					             				     
README					     					 

Download MNIST dataset into ../data/mnist and decompress

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

Download FASHION MNIST dataset into ../data/fashion_mnist and decompress

wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

To run the examples, enter the given folders and run make as follows:

	$ cd /examples/mnist	
	$ make
	$ ./MNIST_Classification
