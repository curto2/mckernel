// Original Source Code by Meroni (https://github.com/Flowx08/)
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

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


