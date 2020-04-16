// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

#ifndef UTIL_HPP
#define UTIL_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <math.h>
#include <random>
#include <limits.h>

////////////////////////////////////////////////////////////
///	NAMESPACE LG
////////////////////////////////////////////////////////////
namespace lg
{
	////////////////////////////////////////////////////////////
	///	UTILITY
	////////////////////////////////////////////////////////////
	namespace util
	{
		//Xorshift 32bit
		uint32_t xorshift32();
		
		//Random number between 0 and 2^32
		int randint();
		
		//Random number between 0 and 1
		float randf();

		//Random number in range (mean - deviation) to (mean + deviation)
		double gaussian(const double mean, const double deviation);

	} //namespace util

} //namespace lg

#endif /* end of include guard: UTIL_HPP */

