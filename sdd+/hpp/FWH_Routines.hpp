/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time.		    

   Authors: Curtó and Zarza.
   c@decurto.tw z@dezarza.tw 						    */

#ifndef FWH_H
#define FWH_H

template <typename DType>
inline void fwh8(DType *data)
{

	const __m128 m3e0 = _mm_loadu_ps((float *)&data[0]);
	const __m128 m3e4 = _mm_loadu_ps((float *)&data[4]);

	const __m128 m3f0 = _mm_add_ps(m3e0, m3e4);
	const __m128 m3f4 = _mm_sub_ps(m3e0, m3e4);

	_mm_storeu_ps((float *)&data[0], m3f0);
	_mm_storeu_ps((float *)&data[4], m3f4);

	float m2g0 = (float)data[0];
	float m2g1 = (float)data[1];
	float m2g2 = (float)data[2];
	float m2g3 = (float)data[3];
	data[0] = m2g0 + m2g2;
	data[1] = m2g1 + m2g3;
	data[2] = m2g0 - m2g2;
	data[3] = m2g1 - m2g3;

	float m2g4 = (float)data[4];
	float m2g5 = (float)data[5];
	float m2g6 = (float)data[6];
	float m2g7 = (float)data[7];
	data[4] = m2g4 + m2g6;
	data[5] = m2g5 + m2g7;
	data[6] = m2g4 - m2g6;
	data[7] = m2g5 - m2g7;

	float m1h0 = (float)data[0];
	float m1h1 = (float)data[1];
	data[0] = m1h0 + m1h1;
	data[1] = m1h0 - m1h1;

	float m1h2 = (float)data[2];
	float m1h3 = (float)data[3];
	data[2] = m1h2 + m1h3;
	data[3] = m1h2 - m1h3;

	float m1h4 = (float)data[4];
	float m1h5 = (float)data[5];
	data[4] = m1h4 + m1h5;
	data[5] = m1h4 - m1h5;

	float m1h6 = (float)data[6];
	float m1h7 = (float)data[7];
	data[6] = m1h6 + m1h7;
	data[7] = m1h6 - m1h7;

}

template <typename DType>
inline void fwh4(DType *data)
{

	float m2g0 = (float)data[0];
	float m2g1 = (float)data[1];
	float m2g2 = (float)data[2];
	float m2g3 = (float)data[3];
	data[0] = m2g0 + m2g2;
	data[1] = m2g1 + m2g3;
	data[2] = m2g0 - m2g2;
	data[3] = m2g1 - m2g3;

	float m1h0 = (float)data[0];
	float m1h1 = (float)data[1];
	data[0] = m1h0 + m1h1;
	data[1] = m1h0 - m1h1;

	float m1h2 = (float)data[2];
	float m1h3 = (float)data[3];
	data[2] = m1h2 + m1h3;
	data[3] = m1h2 - m1h3;

}

template <typename DType>
inline void fwh2(DType *data)
{

	float m1h0 = (float)data[0];
	float m1h1 = (float)data[1];
	data[0] = m1h0 + m1h1;
	data[1] = m1h0 - m1h1;

}

#endif
