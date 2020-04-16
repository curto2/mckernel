// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef BLMNIST_HPP
#define BLMNIST_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <string>

class MNIST_binary_loader
{
public:
	MNIST_binary_loader(const std::string train_images_path, const std::string test_images_path,
											const std::string train_labels_path, const std::string test_labels_path);
	const std::vector< std::vector< unsigned char > >& get_train_images() const;
	const std::vector< std::vector< unsigned char > >& get_test_images() const;
	const std::vector< unsigned char >& get_train_labels() const;
	const std::vector< unsigned char >& get_test_labels() const;

private:
	void load_images(std::ifstream& file, std::vector< std::vector<unsigned char> >& images);
	void load_labels(std::ifstream& file, std::vector<unsigned char>& labels);

	//Data
	std::vector< std::vector< unsigned char > > _train_images;
	std::vector< std::vector< unsigned char > > _test_images;
	std::vector< unsigned char > _train_labels;
	std::vector< unsigned char > _test_labels;
};

#endif /* end of include guard: BLMNIST_HPP */

