// Original Source Code by Meroni,
// https://github.com/Flowx08/artificial_intelligence
// Modified by Curtó & Zarza
// {curto,zarza}.2@my.cityu.edu.hk

#ifndef CSV_HPP
#define CSV_HPP

#include <vector>
#include <string>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	/// \brief	Load comma-separated values file in a 2D vector
	/// from file path
	///
	////////////////////////////////////////////////////////////
	std::vector< std::vector<std::string> > CSV_Load(std::string CSV_filepath);

} /* namespace lg */

#endif /* end of include guard: CSV_HPP */

