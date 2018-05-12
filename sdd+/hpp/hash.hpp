/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time 
   Curtó, Zarza, Yang, Smola, De La Torre, Ngo, and Van Gool 		    

   Authors: Curtó and Zarza
   {curto,zarza}@tinet.cat 						    */

#ifndef PNG_H
#define PNG_H

//Murmurhash is a hash function developed by Austin Appleby
//The author disclaims all copyright to their code. 
unsigned int MurmurHash2 (const void * key, int lt, unsigned int seed);

//PRNG Uniform, Hash Class 
class U_PNG
{
public:
    U_PNG();

    float GetUniform(unsigned long index);
 
    void GetState(unsigned long &seed);
 
    void SetState(unsigned long seed);
 
private:
    unsigned long m_seed;
};

//PRNG Normal and Chi^2, Hash Class
class NC_PNG
{
public:
    NC_PNG();

    float GetNormal(unsigned long index);

    float GetChiSquared(unsigned long index, unsigned long dfreedom);
 
    void GetState(unsigned long &seed1, unsigned long &seed2);
 
    void SetState(unsigned long seed1, unsigned long seed2);
 
private:
    unsigned long m_seed1;
    unsigned long m_seed2;
};

#endif
