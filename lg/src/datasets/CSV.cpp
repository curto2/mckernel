// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "CSV.hpp"
#include <fstream>
#include <sstream>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
		
	////////////////////////////////////////////////////////////
	std::vector< std::vector<std::string> > CSV_Load(std::string CSV_filepath)
	{
		//Create output vector
		std::vector< std::vector<std::string> > data;

		//Open file
		std::ifstream file(CSV_filepath);

		//Check for errors
		if (!file) {
			printf("Error, can't load CSV file of path %s\n", CSV_filepath.c_str());
			return data;
		}

		//Read file line by line
		std::string line;
		std::string cell;
		while (std::getline(file, line)) {
			//Add dimension to data
			data.push_back(std::vector<std::string>());

			//Read all the cells in the line
			std::stringstream lineStream(line);
			while (std::getline(lineStream, cell, ',')) {
				//Store the content of every cell
				data.back().push_back(cell);
			}
		}

		return data;
	}
	
} /* namespace lg */
