// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef FILES_HPP
#define FILES_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include <vector>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	
	////////////////////////////////////////////////////////////
	///	FILES
	////////////////////////////////////////////////////////////
	namespace files
	{
		
		std::vector<std::string> listdir(std::string folderpath);
		std::string get_extension(std::string path);
		std::string remove_extension(std::string path);
		bool exists(std::string filepath);

	} /* namespace files */

} /* namespace lg */

#endif /* end of include guard: FILES_HPP */


