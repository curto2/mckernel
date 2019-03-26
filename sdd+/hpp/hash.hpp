/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk 						    */

#ifndef PN_H
#define PN_H

//Murmurhash is a hash function developed by Austin Appleby
//The author disclaims all copyright to their code. 
unsigned int MurmurHash2 (const void * key, int lt, unsigned int seed);

//PN Uniform, Hash Class 
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

//PN Normal and Chi^2, Hash Class
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
