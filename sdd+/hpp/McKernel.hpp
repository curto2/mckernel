/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time 
   Curtó, Zarza, Yang, Smola, De La Torre, Ngo, and Van Gool 		    

   Authors: Curtó and Zarza
   {curto,zarza}@tinet.cat 						    */

#ifndef MCKERNEL_H
#define MCKERNEL_H

#include <vector>
#include <math.h>
#include <random>
#include <emmintrin.h>
#include "FWHT_Routines.hpp"
#include "hash.hpp"

using namespace std;

//In-place diagonal product 
template <typename Dtype>
inline void dproduct(Dtype* data_in, const Dtype* dl, const unsigned long nv, const unsigned long D, 
  const unsigned long dn, const unsigned long dn_D) {

  for (unsigned long i = 0; i < nv; ++i){
    for(unsigned long k = 0; k < D; ++k){
      for (unsigned long j = 0; j < dn; j += 8){

        unsigned long t = j + k * dn;
        unsigned long t1 = i * dn_D + t;
        unsigned long t2 = t1 + 4;

        const __m128 src0 = _mm_loadu_ps((float *)&data_in[t1]);
        const __m128 src1 = _mm_loadu_ps((float *)&data_in[t2]);
        const __m128 src2 = _mm_loadu_ps((float *)&dl[t]);
        const __m128 src3 = _mm_loadu_ps((float *)&dl[t + 4]);

        const __m128 a0 = _mm_mul_ps(src0, src2);
        const __m128 a1 = _mm_mul_ps(src1, src3);

        _mm_storeu_ps((float *)&data_in[t1], a0);
        _mm_storeu_ps((float *)&data_in[t2], a1);

      }
    }
  }
}

//Diagonal product B (with padding)
template <typename Dtype>
inline void dproductB( Dtype* data_in, vector<long>& dl, const unsigned long nv, const unsigned long dn, 
  const unsigned long dnpg, const unsigned long dn_D, Dtype* data_out)
{
  for(unsigned long i = 0; i < nv; ++i)
  {
    for(unsigned long j = 0; j < dn_D; ++j)
    {
      unsigned long index = j % dnpg;
      if(index < dn){
        if(dl[j] == 1)
          data_out[i * dn_D + j] = data_in[i * dn + index];
        else
          data_out[i * dn_D + j] = (-1) * data_in[i * dn + index];
      }
      else
        data_out[j + i * dn_D] = (Dtype)0;
    }
  }
}

//In-place diagonal product 
template <typename Dtype>
inline void dlproduct(Dtype* data_in, const Dtype* dl, const unsigned long nv, const unsigned long dn)
{
  for (unsigned long i = 0; i < nv; ++i)
    for (unsigned long j = 0; j < dn; ++j)
      data_in[i * dn + j] = dl[j] * data_in[i * dn + j];
}

//In-place Fast Walsh Hadamard Transform
template <typename DType>
inline void fwht(DType* data, unsigned long lgn)
{
  /* The algorithm is based on the Fast Fourier Transform, first 
  it goes down in deep and solves iteratively half of the computation */

  if(lgn < 4)
  {
    switch ( lgn )
    {
      case 3:
        fwht8(data);
        break;
      case 2:
        fwht4(data);
        break;
      case 1:
        fwht2(data);
        break;
    }
    return void();
  }

  for (unsigned long lgs = lgn; lgs > 3; --lgs)
    {

    const unsigned long s = (1UL << lgs);
    const unsigned long hs = (s >> 1);

        for(unsigned long r1 = 0, r2 = hs; r1 < hs; r1 += 8, r2 += 8)
        {
 
            const unsigned long r3 = r1 + 4;
            const unsigned long r4 = r2 + 4;

            const __m128 src0 = _mm_loadu_ps((float *)&data[r1]);
            const __m128 src1 = _mm_loadu_ps((float *)&data[r2]);
            const __m128 src2 = _mm_loadu_ps((float *)&data[r3]);
            const __m128 src3 = _mm_loadu_ps((float *)&data[r4]);

            const __m128 a0 = _mm_add_ps(src0, src1);
            const __m128 a1 = _mm_sub_ps(src0, src1);

            const __m128 a2 = _mm_add_ps(src2, src3);
            const __m128 a3 = _mm_sub_ps(src2, src3);

            _mm_storeu_ps((float *)&data[r1], a0);
            _mm_storeu_ps((float *)&data[r2], a1);

            _mm_storeu_ps((float *)&data[r3], a2);
            _mm_storeu_ps((float *)&data[r4], a3);

        }
    }

  const __m128 src00 = _mm_loadu_ps((float *)&data[0]);
  const __m128 src11 = _mm_loadu_ps((float *)&data[4]);

  const __m128 a00 = _mm_add_ps(src00, src11);
  const __m128 a11 = _mm_sub_ps(src00, src11);

  _mm_storeu_ps((float *)&data[0], a00);
  _mm_storeu_ps((float *)&data[4], a11);

  DType u = data[0];
  DType uu = data[1];
  DType v = data[2];
  DType vv = data[3];
  data[0] = u + v;
  data[1] = uu + vv;
  data[2] = u - v;
  data[3] = uu - vv;

  DType u2 = data[0];
  DType v2 = data[1];
  data[0] = u2 + v2;
  data[1] = u2 - v2;

  /* Here it stops going down and starts going up to compute the remaining
  half vector from bottom to top */

  DType a0, a1;
  a0 = data[2];
  a1 = data[3];
  data[2] = a0 + a1;
  data[3] = a0 - a1; 

  fwht4(data + 4);
  fwht8(data + 8);
 
  /* This for() is intended to solve the remaining computation till (last level) - 1. 
  It computes recursively until it uses an existing FWHT routine. It
  can be adapted to use different FWHT routines, e.g. here it uses length 8 */

    for(unsigned long lgs = 4; lgs < lgn; ++lgs){ 

        unsigned long n = (1UL << lgs);
        DType *tmp = data + n;

        for (unsigned long lg = lgs; lg >= 3; --lg)
        {

		    const unsigned long g = (1UL << lg);
		    const unsigned long hg = (g >> 1);

		    for(unsigned long j = 0; j < n; j += g)
		    {

		        if(lg > 3)
		        {

		            for(unsigned long r1 = j, r2 = j + hg; r1 < j + hg; r1 += 8, r2 += 8)
		            {

			            const unsigned long r3 = r1 + 4;
			            const unsigned long r4 = r2 + 4;

			            const __m128 src0 = _mm_loadu_ps((float *)&tmp[r1]);
			            const __m128 src1 = _mm_loadu_ps((float *)&tmp[r2]);
			            const __m128 src2 = _mm_loadu_ps((float *)&tmp[r3]);
			            const __m128 src3 = _mm_loadu_ps((float *)&tmp[r4]);

			            const __m128 a0 = _mm_add_ps(src0, src1);
			            const __m128 a1 = _mm_sub_ps(src0, src1);

			            const __m128 a2 = _mm_add_ps(src2, src3);
			            const __m128 a3 = _mm_sub_ps(src2, src3);

			            _mm_storeu_ps((float *)&tmp[r1], a0);
			            _mm_storeu_ps((float *)&tmp[r2], a1);

			            _mm_storeu_ps((float *)&tmp[r3], a2);
			            _mm_storeu_ps((float *)&tmp[r4], a3);
		                  
		            }

	            }else{
	                fwht8(tmp + j);
	         	}
	        }
	    }
    }
}

//Fisher Yates Shuffle
template <typename Dtype>
inline void fys(Dtype* data_in, const unsigned long d, U_PNG u_png) 
{

  for (unsigned long i = 0; i < d - 1; ++i)
  {
    unsigned long b = (long) u_png.GetUniform(i) % (d - i) + i;
    Dtype axr = data_in[i];
    data_in[i] = data_in[(unsigned long)b];
    data_in[(unsigned long)b] = axr;      
  }
}

//In-place random permutation (with Fisher Yates Shuffle)
template <typename Dtype>
inline void pn(Dtype* data_in, unsigned long* p, const unsigned long nv, const unsigned long D, 
  const unsigned long dn, const unsigned long dn_D)
{
  Dtype* data_out = new Dtype[dn_D * nv];

  for (unsigned long i = 0; i < nv; ++i)
    for (unsigned long k = 0; k < D; ++k)
      for (unsigned long j = 0; j < dn; ++j)
        data_out[i * dn_D + k * dn + j] = data_in[i * dn_D + k * dn + (unsigned long)(p[j + k * dn])];   

  for (unsigned long i = 0; i < nv; ++i)
    for (unsigned long j = 0; j < dn_D; ++j)
      data_in[i * dn_D + j] = data_out[i * dn_D + j];
    
  delete[] data_out;
}

#endif
