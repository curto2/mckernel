// Original Source Code by Meroni (https://www.github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.tw z@dezarza.tw

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Util.hpp"

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
		uint32_t xorshift32()
		{
			static uint32_t y = 2463534242UL;
			y ^= (y << 13);
			y ^= (y >> 17);
			y ^= (y << 15);
			return y;
		}
		
		//Random number between 0 and 2^32
		int randint()
		{
			return xorshift32();
		}

		//Random number between 0 and 1
		float randf()
		{ 
			return (double)xorshift32() / (double)UINT_MAX; 
		}

		//Random number in range (mean - deviation) to (mean + deviation)
		double gaussian(const double mean, const double deviation)
		{ 
			return (mean - deviation) + randf() * deviation * 2.f; 
		}
		
	} /* namespace util */
	
} /* namespace lg */
