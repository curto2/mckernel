/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time.		    

   Authors: Curtó and Zarza.
   c@decurto.tw z@dezarza.tw 						    */

#ifndef MCKERNEL_H
#define MCKERNEL_H

#include <vector>
#include <math.h>
#include <random>
#include <emmintrin.h>
#include "FWH_Routines.hpp"

using namespace std;

//In-place product diagonal 
template <typename Dtype>
inline void dproduct(Dtype* data_in, const Dtype* dl, const unsigned long nv, const unsigned long D, 
  const unsigned long dn, const unsigned long dn_D) {

  for (unsigned long z = 0; z < nv; ++z){
    for(unsigned long k = 0; k < D; ++k){
      for (unsigned long c = 0; c < dn; c += 8){

        unsigned long t = c + k * dn;
        unsigned long t1 = z * dn_D + t;
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

//Product diagonal B (with padding)
template <typename Dtype>
inline void dproductB( Dtype* data_in, vector<long>& dl, const unsigned long nv, const unsigned long dn, 
  const unsigned long dnpg, const unsigned long dn_D, Dtype* data_out)
{
  for(unsigned long z = 0; z < nv; ++z)
  {
    for(unsigned long c = 0; c < dn_D; ++c)
    {
      unsigned long index = c % dnpg;
      if(index < dn){
        if(dl[c] == 1)
          data_out[z * dn_D + c] = data_in[z * dn + index];
        else
          data_out[z * dn_D + c] = (-1) * data_in[z * dn + index];
      }
      else
        data_out[c + z * dn_D] = (Dtype)0;
    }
  }
}

//In-place product diagonal 
template <typename Dtype>
inline void dlproduct(Dtype* data_in, const Dtype* dl, const unsigned long nv, const unsigned long dn)
{
  for (unsigned long z = 0; z < nv; ++z)
    for (unsigned long c = 0; c < dn; ++c)
      data_in[z * dn + c] = dl[c] * data_in[z * dn + c];
}

//In-place Fast Walsh Hadamard
template <typename DType>
inline void fwh(DType* data, unsigned long lgn)
{
  /* The algorithm is based on the FAST FOURIER Transform, first 
  it goes down in deep and solves iteratively half of the computation */

  if(lgn < 4)
  {
    switch ( lgn )
    {
      case 3:
        fwh8(data);
        break;
      case 2:
        fwh4(data);
        break;
      case 1:
        fwh2(data);
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

  fwh4(data + 4);
  fwh8(data + 8);
 
  /* This for() is intended to solve the remaining computation till (last level) - 1. 
  It computes recursively until it uses an existing FWH routine. It
  can be adapted to use different FWH routines, e.g. here it uses length 8 */

    for(unsigned long lgs = 4; lgs < lgn; ++lgs){ 

        unsigned long n = (1UL << lgs);
        DType *tmp = data + n;

        for (unsigned long lg = lgs; lg >= 3; --lg)
        {

		    const unsigned long g = (1UL << lg);
		    const unsigned long hg = (g >> 1);

		    for(unsigned long c = 0; c < n; c += g)
		    {

		        if(lg > 3)
		        {

		            for(unsigned long r1 = c, r2 = c + hg; r1 < c + hg; r1 += 8, r2 += 8)
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
	                fwh8(tmp + c);
	         	}
	        }
	    }
    }
}

//FISHER YATES Shuffle
template <typename Dtype>
inline void fy(Dtype* data_in, const unsigned long d, mt19937 seed) 
{
  for (unsigned long z = 0; z < d - 1; ++z)
  {
    uniform_int_distribution<> uid(z, d - 1);
    unsigned long b = uid(seed);
    Dtype axr = data_in[z];
    data_in[z] = data_in[(unsigned long)b];
    data_in[(unsigned long)b] = axr;      
  }
}

//In-place random permutation (with FISHER YATES Shuffle)
template <typename Dtype>
inline void pn(Dtype* data_in, unsigned long* p, const unsigned long nv, const unsigned long D, 
  const unsigned long dn, const unsigned long dn_D)
{
  Dtype* data_out = new Dtype[dn_D * nv];

  for (unsigned long z = 0; z < nv; ++z)
    for (unsigned long k = 0; k < D; ++k)
      for (unsigned long c = 0; c < dn; ++c)
        data_out[z * dn_D + k * dn + c] = data_in[z * dn_D + k * dn + (unsigned long)(p[c + k * dn])];   

  for (unsigned long z = 0; z < nv; ++z)
    for (unsigned long c = 0; c < dn_D; ++c)
      data_in[z * dn_D + c] = data_out[z * dn_D + c];
    
  delete[] data_out;
}

#endif
