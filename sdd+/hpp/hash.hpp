/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time.		    

   Authors: Curtó and Zarza.
   c@decurto.tw z@dezarza.tw 						    */

#ifndef PN_H
#define PN_H

//Murmurhash is a function of hashing developed by Appleby.
//The author disclaims all copyright to their code. 
unsigned int MurmurHash2 (const void * key, int lt, unsigned int seed);

//PN Uniform 
class U_PN
{
public:
    U_PN();

    float GetUniform(unsigned long index);
 
    void GetState(unsigned long &seed);
 
    void SetState(unsigned long seed);
 
private:
    unsigned long m_seed;
};

//PN Normal and Chi^2
class NC_PN
{
public:
    NC_PN();

    float GetNormal(unsigned long index);

    float GetChiSquared(unsigned long index, unsigned long dfreedom);
 
    void GetState(unsigned long &seed1, unsigned long &seed2);
 
    void SetState(unsigned long seed1, unsigned long seed2);
 
private:
    unsigned long m_seed1;
    unsigned long m_seed2;
};

#endif
