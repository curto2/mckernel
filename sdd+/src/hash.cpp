/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk 						    */

#include <math.h>
#include "../hpp/hash.hpp"

//Murmurhash is a hash function developed by Austin Appleby
//The author disclaims all copyright to their code. 
unsigned int MurmurHash2 ( const void * key, int lt, unsigned int seed )
{
    //'m' and 'r' are mixing constants generated offline.
    //They're not really 'magic', they just happen to work well.
 
    const unsigned int m = 0x5bd1e995;
    const int r = 24;
 
    //Initialize the hash to a 'random' value
 
    unsigned int h = seed ^ lt;
 
    //Mix 4 bytes at a time into the hash
 
    const unsigned char * data = (const unsigned char *)key;
 
    while(lt >= 4)
    {
        unsigned int k = *(unsigned int *)data;
 
        k *= m;
        k ^= k >> r;
        k *= m;
         
        h *= m;
        h ^= k;
 
        data += 4;
        lt -= 4;
    }
     
    //Handle the last few bytes of the input array
 
    switch(lt)
    {
    case 3: h ^= data[2] << 16;
    case 2: h ^= data[1] << 8;
    case 1: h ^= data[0];
            h *= m;
    };
 
    //Do a few final mixes of the hash to ensure the last few
    //bytes are well-incorporated.
 
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
 
    return h;
}

//PN Uniform, Hash Class 
U_PN::U_PN()
{
    m_seed = 0;
}

float U_PN::GetUniform(unsigned long index)
{
    return MurmurHash2(&index, sizeof(index), m_seed) / (float)( 1UL << 32 );
}

void U_PN::GetState(unsigned long &seed)
{
    seed = m_seed;
}

void U_PN::SetState(unsigned long seed)
{
    m_seed = seed;
}

//PN Normal and Chi^2, Hash Class 
NC_PN::NC_PN()
{
    m_seed1 = 0;
    m_seed2 = 0;
}

float NC_PN::GetNormal(unsigned long index)
{
      float a = MurmurHash2(&index, sizeof(index), m_seed1) / (float)( 1UL << 32 );
      float b = MurmurHash2(&index, sizeof(index), m_seed2) / (float)( 1UL << 32 );

      return cos(2. * M_PI * b) * sqrt(-2. * log(a));
}

float NC_PN::GetChiSquared(unsigned long index, unsigned long dfreedom)
{
      float a = MurmurHash2(&index, sizeof(index), m_seed1) / (float)( 1UL << 32 );
      float b = MurmurHash2(&index, sizeof(index), m_seed2) / (float)( 1UL << 32 );

      //Approximation Wilson and Hilferty
      return dfreedom * pow( sqrt(2./ (9 * dfreedom)) * cos(2. * M_PI * b) * sqrt(-2. * log(a)) + (1 - 2./(9 * dfreedom)) , 3 );
}

void NC_PN::GetState(unsigned long &seed1, unsigned long &seed2)
{
    seed1 = m_seed1;
    seed2 = m_seed2;
}

void NC_PN::SetState(unsigned long seed1, unsigned long seed2)
{
    m_seed1 = seed1;
    m_seed2 = seed2;
}
 
