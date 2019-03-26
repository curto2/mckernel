/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk 						    */    

#include <iostream>
#include <iterator>
#include <fstream>
#include <cxxtest/TestSuite.h>
#include "../hpp/McKernel.hpp"

// Unit Test McKernel
class TestSuite_McKernel : public CxxTest::TestSuite 
{
	public:
		void test_trivial_hadamard2( void )
		{
			unsigned long lt = 1UL << 1;

			vector<float> data_in(lt);
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			TS_ASSERT_EQUALS(data_in[0], data_out[0]);
			TS_ASSERT_EQUALS(data_in[1], data_out[1]);
		}

		void test_hadamard2( void )
		{
			unsigned long lt = 1UL << 1;

			vector<float> data_in = {3, 8};
			vector<float> data_out = {11, -5};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			TS_ASSERT_EQUALS(data_in[0], data_out[0]);
			TS_ASSERT_EQUALS(data_in[1], data_out[1]);
		}

		void test_float_hadamard2( void )
		{
			unsigned long lt = 1UL << 1;
			
			vector<float> data_in = {12.4399, -9.18444};
			vector<float> data_out = {3.25544, 21.6243};
			
			float* data = &data_in[0];

			fwh(data, log2(lt));
  
			TS_ASSERT_DELTA(data_in[0], data_out[0], 0.01);
			TS_ASSERT_DELTA(data_in[1], data_out[1], 0.01);
		}

		void test_trivial_hadamard4( void )
		{
			unsigned long lt = 1UL << 2;

			vector<float> data_in(lt);
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);

		}

		void test_hadamard4( void )
		{
			unsigned long lt = 1UL << 2;

			vector<float> data_in = {3, 8, 2, 7};
			vector<float> data_out = {20, -10, 2, 0};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);

		}

		void test_float_hadamard4( void )
		{
			unsigned long lt = 1UL << 2;

			vector<float> data_in = {12.4399, -9.18444, 11.6708, 13.2429};
			vector<float> data_out = {28.1691, 20.0522, -21.6583, 23.1965};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01);

		}

		void test_trivial_hadamard8( void )
		{
			unsigned long lt = 1UL << 3;

			vector<float> data_in(lt);
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);

		}

		void test_hadamard8( void )
		{
			unsigned long lt = 1UL << 3;

			vector<float> data_in = {3, 8, 2, 7, 7, 3, 6, 3};
			vector<float> data_out = {39, -3, 3, 1, 1, -17, 1, -1};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);

		}

		void test_float_hadamard8( void )
		{
			unsigned long lt = 1UL << 3;

			vector<float> data_in = {12.4399, -9.18444, 11.6708, 13.2429, 0.943871, -18.9393, -6.57406, -13.2826};
			vector<float> data_out = {-9.68293, 46.6438, -19.7971, 36.3712, 66.0212, -6.53954, -23.5195, 10.0218};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01);

		}

		void test_trivial_hadamard16(void)
		{
			unsigned long lt = 1UL << 4;

			vector<float> data_in(lt); 
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c] , data_out[c]);

		}

		void test_hadamard16(void)
		{
			unsigned long lt = 1UL << 4;

			vector<float> data_in = {1, 8, 1, 0, 5, 3, 0, 0, 4, 3, 7, 8, 5, 0, 8, 0};
			vector<float> data_out = {53, 9, 5, -7, 11, -21, -5, -5, -17, -17, 27, -5, -7, 5, 5, -15};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c] , data_out[c]);

		}

		void test_float_hadamard16(void)
		{
			unsigned long lt = 1UL << 4;

			vector<float> data_in = {12.4399, -9.18444, 11.6708, 13.2429, 0.943871, -18.9393, -6.57406, -13.2826, -19.777, 14.7397,
										-9.3895, 4.44935, -16.502, -12.2115, 1.59443, -10.8866};
			vector<float> data_out = {-57.6659, 6.47872, -39.3155, -1.07824, 94.0495, -63.0856, -4.19521, 6.11552, 38.3, 86.809,
										-0.278664, 73.8206, 37.993, 50.0065, -42.8437, 13.9282};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01);

		}

		void test_trivial_hadamard32(void)
		{
			unsigned long lt = 1UL << 5;

			vector<float> data_in(lt); 
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c] , data_out[c]);

		}

		void test_hadamard32(void)
		{
			unsigned long lt = 1UL << 5;

			vector<float> data_in = {1, 8, 1, 0, 5, 3, 0, 0, 4, 3, 7, 8, 5, 0, 8, 0, 2, 0, 7, 4, 0, 8, 8, 8, 3, 0, 0, 4, 8, 2, 8, 0};
			vector<float> data_out = {115, 19, -11, -11, -11, -23, -9, 11, -5, -33, 9, -19, -7, 33, 7, -17, -9, -1, 21, -3, 33, -19,
										-1, -21, -29, -1, 45, 9, -7, -23, 3, -13};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);

		}

		void test_float_hadamard32(void)
		{

			unsigned long lt =  1UL << 5;

			vector<float> data_in = {12.4399, -9.18444, 11.6708, 13.2429, 0.943871, -18.9393, -6.57406, -13.2826, -19.777, 14.7397,
										-9.3895, 4.44935, -16.502, -12.2115, 1.59443, -10.8866, -7.67366, -13.4956, 18.9838, -18.5481,
										2.85637, -16.9383, -4.94062, 7.59839, 4.14646, 11.9015, 1.62259, 8.4356, -1.21352, 6.63496,
										-5.3924, -8.77364};

			vector<float> data_out = {-72.4621, 38.0529, -52.0828, -12.6263, 119.591, -37.0883, -22.6581, -47.6405, -11.2193, 156.454,
										-64.2006, 86.6161, -6.16719, 96.2055, -34.1113, -60.4032, -42.8697, -25.0955, -26.5481, 10.4698,
										68.508, -89.0829, 14.2677, 59.8716, 87.8194, 17.1641, 63.6433, 61.025, 82.1532, 3.80748, -51.5762,
										88.2595};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01); 

		}

		void test_trivial_hadamard64(void)
		{
			unsigned long lt = 1UL << 6;

			vector<float> data_in(lt); 
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c] , data_out[c]);

		}

		void test_hadamard64(void)
		{

			unsigned long lt =  1UL << 6;

			vector<float> data_in = {1, 8, 1, 0, 5, 3, 0, 0, 4, 3, 7, 8, 5, 0, 8, 0, 2, 0, 7, 4, 0, 8, 8, 8, 3, 0, 0, 4, 8, 2, 8, 0, 
			 			1, 0, 0, 5, 1, 7, 5, 5, 1, 3, 3, 4, 2, 0, 3, 4, 0, 1, 7, 7, 0, 6, 6, 1, 4, 4, 3, 3, 6, 2, 2, 6};

			vector<float> data_out = {217, 5, -37, -13, -21, -25, -19, 21, -3, -43, -5, -41, -17, 39, -11, 17, -23, -11, 19, 3, 23, -21, 
			 				9, -15, -23, -7, 63, 27, -25, -17, 13, -15, 13, 33, 15, -9, -1, -21, 1, 1, -7, -23, 23, 3, 3, 27, 
							25, -51, 5, 9, 23, -9, 43, -17, -11, -27, -35, 5, 27, -9, 11, -29, -7, -11};

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]); 

		}


		void test_float_hadamard64(void)
		{

			unsigned long lt =  1UL << 6;

			vector<float> data_in = {12.4399, -9.18444, 11.6708, 13.2429, 0.943871, -18.9393, -6.57406, -13.2826, -19.777, 14.7397,
										-9.3895, 4.44935, -16.502, -12.2115, 1.59443, -10.8866, -7.67366, -13.4956, 18.9838, -18.5481,
										2.85637, -16.9383, -4.94062, 7.59839, 4.14646, 11.9015, 1.62259, 8.4356, -1.21352, 6.63496,
										-5.3924, -8.77364, 17.4505, -13.7216, -15.5307, -1.60562, -12.6609, -2.10476, 5.11181, -12.4379,
										-7.36501, 15.7223, 12.0114, -3.86702, -16.4892, -6.39413, 5.24641, -4.16281, 0.110283, 4.2302,
										-2.71093, -17.0333, 7.29188, 12.3484, 10.565, -8.56166, 4.24998, -7.81237, 19.8739, -16.9635,
										18.8226, -5.51847, -5.73718, 16.2731}; 
			
			vector<float> data_out = {-83.8313, 129.902, -24.3941, -75.1544, 105.037, 9.3436, 27.0333, -59.2844, -58.3687, 161.631,
										18.4053, 97.8757, -48.3397, 62.5454, -14.2974, 109.129, -113.095, -84.2528, -74.8937, -9.95949,
										177.033, -128.054, 69.5454, 178.999, 74.5663, 76.2573, 55.3966, 200.688, 65.0916, 71.0208,
										-63.1399, 104.057, -61.0928, -53.7963, -79.7716, 49.9018, 134.145, -83.5201, -72.3495, -35.9966,
										35.9301, 151.277, -146.806, 75.3566, 36.0053, 129.866, -53.9251, -229.935, 27.3555, 34.0618,
										21.7974, 30.8991, -40.0172, -50.1119, -41.01, -59.2556, 101.073, -41.9291, 71.8899, -78.6381,
										99.2147, -63.4058, -40.0124, 72.4622}; 
			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01); 

		}

		void test_trivial_hadamard128(void)
		{
			unsigned long lt = 1UL << 7;

			vector<float> data_in(lt); 
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c] , data_out[c]);

		}

		void test_hadamard128(void)
		{

			unsigned long lt =  1UL << 7;

			vector<float> data_in = {1, 8, 1, 0, 5, 3, 0, 0, 4, 3, 7, 8, 5, 0, 8, 0, 2, 0, 7, 4, 0, 8, 8, 8, 3, 0, 0, 4, 8, 2, 8, 0,
										1, 0, 0, 5, 1, 7, 5, 5, 1, 3, 3, 4, 2, 0, 3, 4, 0, 1, 7, 7, 0, 6, 6, 1, 4, 4, 3, 3, 6, 2, 2,
										6, 3, 0, 2, 2, 8, 5, 6, 7, 7, 0, 3, 0, 7, 6, 4, 7, 5, 0, 5, 5, 4, 0, 4, 0, 4, 7, 1, 0, 1, 1,
										4, 2, 2, 6, 2, 8, 0, 8, 6, 7, 6, 7, 7, 4, 2, 1, 2, 7, 1, 5, 1, 4, 5, 5, 4, 8, 2, 3, 8, 3, 3,
										1, 3, 5}; 
			
			vector<float> data_out = {463, 3, -53, -1, -41, -13, 3, 7, 7, -65, -17, -33, -49, 15, -33, 47, 15, -25, 23, 15, -5, 3, 11,
										-5, -29, -25, 55, 3, -33, -29, 19, -9, -13, 87, 43, 7, -25, -21, -1, 19, -13, 19, -1, 15, 31,
										31, 3, -69, 15, 23, 27, 11, -17, -1, -29, -45, -37, -9, 43, -9, 3, -41, 19, -33, -29, 7, -21,
										-25, -1, -37, -41, 35, -13, -21, 7, -49, 15, 63, 11, -13, -61, 3, 15, -9, 51, -45, 7, -25, -17,
										11, 71, 51, -17, -5, 7, -21, 39, -21, -13, -25, 23, -21, 3, -17, -1, -65, 47, -9, -25, 23, 47,
										-33, -5, -5, 19, -29, 103, -33, 7, -9, -33, 19, 11, -9, 19, -17, -33, 11}; 
			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]); 

		}


		void test_float_hadamard128(void)
		{

			unsigned long lt =  1UL << 7;

			vector<float> data_in = {12.4399, -9.18444, 11.6708, 13.2429, 0.943871, -18.9393, -6.57406, -13.2826, -19.777, 14.7397,
										-9.3895, 4.44935, -16.502, -12.2115, 1.59443, -10.8866, -7.67366, -13.4956, 18.9838, -18.5481,
										2.85637, -16.9383, -4.94062, 7.59839, 4.14646, 11.9015, 1.62259, 8.4356, -1.21352, 6.63496,
										-5.3924, -8.77364, 17.4505, -13.7216, -15.5307, -1.60562, -12.6609, -2.10476, 5.11181, -12.4379,
										-7.36501, 15.7223, 12.0114, -3.86702, -16.4892, -6.39413, 5.24641, -4.16281, 0.110283, 4.2302,
										-2.71093, -17.0333, 7.29188, 12.3484, 10.565, -8.56166, 4.24998, -7.81237, 19.8739, -16.9635,
										18.8226, -5.51847, -5.73718, 16.2731, 0.7599, -1.26788, -5.33252, 8.09896, 16.6274, 19.7793, 15.661,
										-10.7377, 15.5016, 7.67249, 5.39532, 19.0124, -18.7216, -9.35827, -5.15038, 1.38865, 14.8719, 12.1387,
										4.3553, 2.16381, 4.48713, -5.07965, 13.6021, -11.2629, 7.10798, 13.4761, -8.22643, 5.93057, -12.0424,
										6.03639, 2.20367, 8.71752, -15.2315, 16.8712, -3.18353, -18.6041, 16.6504, -7.52248, -9.34179, 12.152,
										-19.85, 16.0535, 11.1645, -18.5716, -13.3047, -13.9859, 2.81702, -18.4328, 18.1528, -12.8277, 3.73101,
										2.63993, 2.09267, -2.66684, 11.377, -10.7993, -9.19075, -16.8494, 15.1312, -1.23314, 9.18701, -2.6651,
										-12.5156, 13.9555}; 

			
			vector<float> data_out = {-20.8232, 184.466, -13.5977, -135.131, 181.75, -44.3139, 66.0715, -92.7938, 43.3327, 299.476,
										154.241, 121.389, -180.417, -40.5632, -41.5731, 172.571, -182.085, -154.99, -37.0144, -117.051,
										180.257, -218.388, 71.9511, 74.5471, 120.362, -54.0535, 102.701, 280.452, -167.399, 102.074,
										-147.901, -14.1445, 111.516, -139.577, -38.2291, 81.8745, 228.447, -129.814, -8.27322, 206.461,
										14.0714, 249.447, -151.907, -35.5439, -32.0871, 76.2953, -175.101, -258.926, 98.0434, 96.6264,
										-57.7928, 184.117, -52.8493, 42.0287, -193.223, 113.369, 86.8273, -13.4999, 108.893, -201.868,
										57.8951, -116.84, -38.1686, 239.17, -146.84, 75.3385, -35.1905, -15.1776, 28.3239, 63.0011,
										-12.0049, -25.775, -160.07, 23.7864, -117.43, 74.3626, 83.738, 165.654, 12.9782, 45.6872,
										-44.1045, -13.5158, -112.773, 97.1323, 173.81, -37.7201, 67.1396, 283.45, 28.7706, 206.568,
										8.0922, 120.924, 297.582, 39.9674, 21.6206, 222.258, -233.702, 31.9843, -121.314, 17.9291,
										39.842, -37.2263, -136.426, -278.455, 57.7888, 53.1057, -141.706, 186.257, 104.098, 183.436,
										67.2512, -200.945, -43.3325, -28.5028, 101.388, -122.319, -27.1852, -142.253, 111.203, -231.88,
										115.318, -70.3583, 34.8869, 44.5915, 140.534, -9.97193, -41.8563, -94.2452}; 
			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01); 

		}

		void test_trivial_hadamard256(void)
		{
			unsigned long lt = 1UL << 8;

			vector<float> data_in(lt); 
			vector<float> data_out(lt);

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c] , data_out[c]);

		}

		void test_hadamard256(void)
		{

			unsigned long lt =  1UL << 8;

			vector<float> data_in = {1, 8, 1, 0, 5, 3, 0, 0, 4, 3, 7, 8, 5, 0, 8, 0, 2, 0, 7, 4, 0, 8, 8, 8, 3, 0, 0, 4, 8, 2, 8, 0,
										1, 0, 0, 5, 1, 7, 5, 5, 1, 3, 3, 4, 2, 0, 3, 4, 0, 1, 7, 7, 0, 6, 6, 1, 4, 4, 3, 3, 6, 2, 2,
										6, 3, 0, 2, 2, 8, 5, 6, 7, 7, 0, 3, 0, 7, 6, 4, 7, 5, 0, 5, 5, 4, 0, 4, 0, 4, 7, 1, 0, 1, 1,
										4, 2, 2, 6, 2, 8, 0, 8, 6, 7, 6, 7, 7, 4, 2, 1, 2, 7, 1, 5, 1, 4, 5, 5, 4, 8, 2, 3, 8, 3, 3,
										1, 3, 5, 7, 5, 4, 7, 5, 8, 4, 2, 5, 2, 5, 5, 3, 7, 4, 5, 4, 3, 0, 7, 7, 2, 4, 0, 3, 1, 1, 6,
										2, 4, 0, 7, 7, 2, 4, 3, 2, 8, 4, 5, 1, 0, 1, 5, 5, 3, 8, 7, 5, 6, 4, 3, 6, 8, 1, 0, 1, 2, 7,
										3, 4, 7, 0, 2, 1, 4, 4, 1, 3, 6, 4, 4, 4, 5, 7, 0, 7, 6, 6, 3, 1, 1, 4, 7, 7, 5, 8, 8, 7, 4,
										1, 0, 0, 1, 2, 8, 5, 6, 7, 6, 1, 2, 1, 5, 6, 7, 4, 2, 2, 1, 5, 4, 0, 0, 2, 7, 3, 8, 5, 1, 1,
										6, 8, 2, 5, 1, 8, 8}; 

			
			vector<float> data_out = {974, -14, -62, -6, -74, 10, -18, 22, 24, -80, 16, -36, -36, -12, -88, 100, 48, -8, 32, -12, 12,
										28, 40, 4, -42, -54, 50, 2, 6, 34, 46, -42, -8, 84, 56, 40, -44, -24, 8, 32, 2, 42, -34, 10,
										2, -30, 2, -90, 38, 38, 66, -66, -26, 6, -98, -54, -76, -40, 68, 4, 0, -44, 44, -4, -34, -10,
										22, 10, 2, -30, -98, 90, 12, 24, 72, -64, 52, 56, 68, -28, -28, 0, -40, 8, -4, -40, 28, 12, -2,
										6, 78, 42, -30, 10, -34, 10, 44, -36, -32, -4, 76, -52, 4, 0, -22, -58, 58, 2, -10, 2, 26, -14,
										14, -14, 46, -26, 162, -2, 14, -18, 16, 24, -12, 0, 8, -40, -40, -20, -48, 20, -44, 4, -8, -36,
										24, -8, -10, -50, -50, -30, -62, 42, 22, -6, -18, -42, 14, 42, -22, -22, -18, -14, -16, 4, 60, 
										4, -72, -92, -8, 24, -18, 90, 30, -26, -6, -18, -10, 6, -28, -4, 32, 20, 60, 92, 4, -48, -8, 8,
										-12, 88, -8, -8, 40, -36, 2, 22, 18, -22, 6, -38, -6, -62, -24, 24, -64, -60, -4, -44, 16, -20,
										-38, -66, -58, -34, -22, 70, -46, 2, -94, 6, 70, -26, 106, -50, -14, -62, -32, 16, 64, 60, -4, 
										-20, 48, -52, 34, -6, 6, -46, -30, 10, 2, -34, 20, -72, 36, -20, -40, 44, 68, -52, -24, 4, -8,
										-32, 44, -64, 0, 0, -82, 14, 34, -18, 30, 6, -26, 42}; 
			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]); 

		}

		void test_float_hadamard256(void)
		{

			unsigned long lt =  1UL << 8;

			vector<float> data_in = {-1.70976, 16.875, 19.7371, -19.3772, 2.21223, -6.39206, -1.33033, 10.4053, 11.1094, 13.3355,
										3.75801, -1.84295, 16.6916, 17.3668, -8.29814, -4.97828, -18.7598, -13.5749, -10.5338, 11.9452,
										-10.9098, 3.02969, 13.1933, 1.87919, -11.5684, -3.83862, 5.32715, -8.21127, -19.3411, 8.41846,
										11.6241, -1.05089, 5.29341, 11.3612, -0.428139, -12.4944, -15.0308, 18.2415, 17.9109, 16.0785,
										11.5771, 1.66895, -5.76443, 8.26863, -0.964204, 5.93743, -16.7096, 0.276019, 12.3625, -7.24345,
										-7.77879, -18.5473, 15.7862, -14.5855, 3.33188, -15.7821, 1.57588, -11.341, -3.9934, 2.23474,
										17.0775, -12.3693, -18.8162, 2.3709, 18.9919, 0.755708, 9.87655, -16.0389, -1.00276, 7.78749,
										-19.9604, -9.4257, -10.5436, -5.72482, 18.8429, 8.49224, -19.7874, -17.8667, -11.2317, 12.5751,
										-5.11016, 0.989464, 14.0278, -9.32392, 6.40396, -2.64028, -5.10605, -12.0202, 6.01875, 10.9005,
										10.2146, 3.09624, 18.5312, 11.3984, -14.5329, 17.5232, -7.84586, 15.3437, -18.5157, 11.1514,
										3.13118, -18.4761, -18.2743, 12.5876, -4.20095, -19.4314, 1.07986, -3.98834, -17.2981, 9.84811,
										-11.4132, -2.40826, -9.16242, -17.3854, 8.26782, 17.2415, -0.0256364, -16.8382, -14.7786,
										-14.0069, 14.0623, 15.436, 9.08936, 12.5936, 6.8344, 14.5565, 10.1167, 18.9885, 9.9002, 11.601,
										10.1399, -6.96861, 13.1248, 11.8656, -14.381, -11.0761, 12.4342, 6.69887, 4.93556, 15.1361,
										-3.45301, 13.5224, -7.27215, 7.38456, 16.137, -19.0043, 4.62611, -3.88863, -15.8426, 9.84749,
										2.10448, 18.2197, 5.28346, -8.80616, 10.8133, -7.88215, -14.2496, 0.930022, -8.89361, 15.6506,
										-7.469, -18.7537, -11.3181, -14.3442, 13.1119, -5.69905, -5.42027, 5.54612, -19.0002, 19.5153,
										0.682238, -2.45319, 13.0376, 13.4101, -15.0686, 9.17465, 14.4058, 9.55748, -14.714, 18.5632,
										-0.595034, 7.3905, 16.7829, -15.3116, 18.5843, 7.59625, -3.19372, -15.6653, -11.4737, 7.91267,
										19.9853, 1.05727, 9.15898, -11.3328, 6.7131, 2.27089, 2.96815, -18.7072, -12.183, 3.96797,
										-19.1919, 8.49925, -18.4852, 13.8458, 1.90934, -13.5539, 3.02043, -3.68489, 16.0036, 8.30644,
										-5.1217, -4.59141, -4.30306, -8.33875, 0.0970125, -5.71871, 19.2575, 16.9033, -1.38402,
										-12.2162, 4.81596, -1.39876, 8.84103, -6.02506, 7.26843, -4.44586, 16.2458, -9.76342, -3.15302,
										-15.9372, 14.2046, -2.3449, 12.5621, 15.7193, -8.49912, -5.52856, -17.8345, 14.5213, 10.7865,
										18.1691, 2.82775, -14.3351, -6.42231, 18.5247, -2.6739, 13.6747, -7.19402, -3.4164, 10.578,
										11.422, 4.36736, -4.60605, -9.9768, -6.7916, 9.36889, 17.2916}; 

			
			vector<float> data_out = {186.508, -63.2127, 145.26, 85.6513, 166.137, 364.259, 174.488, -92.4455, -334.166, 408.044,
										-3.87586, 99.7473, 22.1912, -133.005, -108.487, 79.5883, -75.4212, -250.364, 117.859,
										-18.2998, 64.3173, 66.4556, -58.9906, -358.849, 207.993, 256.354, 100.142, -79.0174, -62.5301,
										233.064, 338.455, -403.022, 14.0537, 80.0824, -13.6954, -153.212, 185.011, 194.353, -137.803,
										-216.851, 95.558, 6.42984, 137.001, -93.9659, 6.46708, 306.525, 92.9921, -72.0431, 123.297,
										198.609, 122.48, 185.404, 311.236, -194.642, -191.022, 4.13102, -70.0942, 52.4224, 304.645,
										-55.6142, 128.93, 154.997, -253.251, -1.41121, 71.8253, 6.38357, 211.656, -169.367, -334.185,
										101.895, -15.2606, 122.906, 209.045, -52.9104, -113.982, -174.329, -262.579, -37.9771, 321.083,
										84.2468, 462.401, -81.8123, 411.275, -66.4945, 35.2486, -18.3941, -212.119, -270.98, -398.488,
										-2.68745, -261.51, -212.349, -85.5801, 103.856, -51.5636, -9.12791, 142.925, -225.695, -335.064,
										-127.158, -1.66266, 257.398, -42.3898, 147.652, -120.994, -140.546, -157.658, 192.537, 68.1769,
										-241.149, -221.715, -234.361, 203.647, 288.352, 533.918, -81.8813, 136.859, -166.115, 165.032,
										-84.6008, -4.07254, 12.7191, -73.2025, -157.321, 194.269, -333.689, 135.629, -317.432, -152.338,
										-37.6331, -59.6638, 10.6184, 34.1846, 24.4732, -332.841, -260, -36.8701, -96.223, 30.1237,
										-267.979, -42.5238, -170.421, 330.772, -160.444, 80.5058, -88.8241, 87.3197, -117.992, 173.003,
										328.074, 359.36, 223.893, 375.294, -490.763, 123.905, -13.4161, -78.6694, -65.1076, 118.648,
										39.4105, 116.981, -9.76563, -60.738, -389.276, -165.45, 17.2164, 30.2455, -54.8897, -90.5768,
										73.4487, -22.4284, 79.8182, -24.8435, -96.2563, -231.036, 165.113, -27.1705, 342.381, 68.339,
										63.0281, 29.7376, 205.947, 63.6689, -324.91, -231.089, 294.616, -58.5715, -28.1983, 1.86747,
										158.27, 89.5894, 33.3265, -47.2144, 168.733, -17.6997, -93.0774, -89.5399, -237.01, 332.653,
										110.028, 144.53, -0.959125, -281.847, -118.963, -60.974, 223.384, 60.9265, -51.4057, 376.007,
										53.3234, -77.1076, -125.082, 30.4392, 76.1004, 176.277, 63.9846, -87.9713, 143.86, -406.271,
										-189.219, 125.652, 117.945, -18.7457, -94.8749, -131.216, -101.19, -24.4724, -61.8112, -113.289,
										-507.499, 2.38537, 29.2264, -121.022, -221.241, -315.287, 455.667, 173.109, -88.2256, 20.254,
										-164.182, -263.797, -10.9189, 32.9783, 378.282, -202.719, -16.0176, -261.603, -454.468, 35.3774,
										45.7354, -94.2102, 47.7759, -22.1315, 128.448, 222.291, 25.5424}; 

			float* data = &data_in[0];

			fwh(data, log2(lt));

			for(unsigned long c = 0; c < lt; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01); 

		}
		
		void test_dproduct1( void )
		{
			unsigned long dn = 1UL << 4;
			unsigned long D = 2;
			unsigned long nv = 1;
			unsigned long dn_D = D * dn;

			vector<float> dt = {6, 1, 7, 3, 3, 4, 7, 6, 5, 1, 5, 2, 6, 7, 7, 6, 6, 7, 0, 1, 6, 3, 3, 0, 3, 4, 0, 3, 3, 6, 5, 0};
			vector<float> data_in =   {6, 1, 7, 3, 3, 4, 7, 6, 5, 1, 5, 2, 6, 7, 7, 6, 6, 7, 0, 1, 6, 3, 3, 0, 3, 4, 0, 3, 3, 6, 5, 0};
			vector<float> data_out = {36, 1, 49, 9, 9, 16, 49, 36, 25, 1, 25, 4, 36, 49, 49, 36, 36, 49, 0, 1, 36, 9, 9, 0, 9, 16, 0, 
										9, 9, 36, 25, 0};

			float* data = &data_in[0];
			float* dl = &dt[0];

			dproduct(data, dl, nv, D, dn, dn_D);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);
		}

		void test_dproduct2( void )
		{
			unsigned long dn = 1UL << 4;
			unsigned long D = 3;
			unsigned long nv = 2;
			unsigned long dn_D = D * dn;

			vector<float> dt = {3, 8, 8, 4, 4, 0, 3, 2, 8, 6, 7, 0, 8, 3, 4, 0, 1, 6, 7, 6, 3, 5, 5, 4, 1, 2, 7, 8, 7, 7, 0, 8, 
										6, 8, 3, 8, 8, 5, 8, 5, 2, 5, 4, 1, 8, 8, 8, 0};
			vector<float> data_in = {3, 8, 8, 4, 4, 0, 3, 2, 8, 6, 7, 0, 8, 3, 4, 0, 1, 6, 7, 6, 3, 5, 5, 4, 1, 2, 7, 8, 7, 7, 0, 8, 6, 
										8, 3, 8, 8, 5, 8, 5, 2, 5, 4, 1, 8, 8, 8, 0, 3, 7, 5, 4, 1, 1, 6, 0, 1, 4, 7, 7, 0, 7, 6, 4, 4, 
										8, 4, 2, 4, 3, 5, 6, 8, 0, 6, 6, 6, 5, 4, 8, 1, 0, 3, 2, 8, 1, 1, 1, 3, 8, 8, 2, 4, 3, 4, 8};

			vector<float> data_out = {9, 64, 64, 16, 16, 0, 9, 4, 64, 36, 49, 0, 64, 9, 16, 0, 1, 36, 49, 36, 9, 25, 25, 16, 1, 4, 49, 
										64, 49, 49, 0, 64, 36, 64, 9, 64, 64, 25, 64, 25, 4, 25, 16, 1, 64, 64, 64, 0, 9, 56, 40, 16, 4,
										0, 18, 0, 8, 24, 49, 0, 0, 21, 24, 0, 4, 48, 28, 12, 12, 15, 25, 24, 8, 0, 42, 48, 42, 35, 0, 64, 
										6, 0, 9, 16, 64, 5, 8, 5, 6, 40, 32, 2, 32, 24, 32, 0};

			float* data = &data_in[0];
			float* dl = &dt[0];

			dproduct(data, dl, nv, D, dn, dn_D);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);
		}

		void test_dproduct3( void )
		{
			unsigned long dn = 1UL << 5;
			unsigned long D = 3;
			unsigned long nv = 4;
			unsigned long dn_D = D * dn;

			vector<float> dt = {16.6057, 7.04453, -14.8742, -12.8921, 0.415212, 3.32884, -3.36608, -15.1568, -15.4119, 5.92053, 
										9.67112, -16.9876, 17.0985, 7.99208, 10.9158, -18.0899, -5.18764, -7.48445, 0.483245, 12.4652, 
										-7.16795, -8.92902, -15.056, 10.7202, 9.61883, -10.3386, 16.5878, 11.0511, 12.8294, 1.20821, 
										16.0874, 9.43505, -11.7473, -18.7868, 16.543, 8.66796, 4.54205, -6.82308, 13.5112, 9.13017, 
										19.0975, 3.18233, 12.1425, 16.1959, -8.82559, 3.05837, 18.1061, 5.98677, 15.5739, -1.4107, 
										-1.54807, -11.594, 9.66028, 3.3959, 19.1262, -0.720892, 13.0573, 15.714, -9.66982, 5.88668, 
										-3.07781, -13.5824, -4.67826, 5.17494, -12.3692, -8.13526, -6.1571, 12.1728, 5.04166, -12.6459, 
										1.30302, 4.13911, 10.5364, -6.55445, 0.335055, -18.2892, 16.5039, -1.55889, 7.69761, 12.0778, 
										17.0304, -13.8505, -19.5162, 6.69068, 9.54545, 19.61, -14.0302, 2.60278, 15.324, -3.70003, 
										-11.5105, -7.7538, 2.71755, 3.81121, 17.4211, 10.3483};

			vector<float> data_in = {16.6057, 7.04453, -14.8742, -12.8921, 0.415212, 3.32884, -3.36608, -15.1568, -15.4119, 5.92053, 
										9.67112, -16.9876, 17.0985, 7.99208, 10.9158, -18.0899, -5.18764, -7.48445, 0.483245, 12.4652, 
										-7.16795, -8.92902, -15.056, 10.7202, 9.61883, -10.3386, 16.5878, 11.0511, 12.8294, 1.20821, 
										16.0874, 9.43505, -11.7473, -18.7868, 16.543, 8.66796, 4.54205, -6.82308, 13.5112, 9.13017, 
										19.0975, 3.18233, 12.1425, 16.1959, -8.82559, 3.05837, 18.1061, 5.98677, 15.5739, -1.4107, 
										-1.54807, -11.594, 9.66028, 3.3959, 19.1262, -0.720892, 13.0573, 15.714, -9.66982, 5.88668, 
										-3.07781, -13.5824, -4.67826, 5.17494, -12.3692, -8.13526, -6.1571, 12.1728, 5.04166, -12.6459, 
										1.30302, 4.13911, 10.5364, -6.55445, 0.335055, -18.2892, 16.5039, -1.55889, 7.69761, 12.0778, 
										17.0304, -13.8505, -19.5162, 6.69068, 9.54545, 19.61, -14.0302, 2.60278, 15.324, -3.70003, 
										-11.5105, -7.7538, 2.71755, 3.81121, 17.4211, 10.3483, 15.6759, -8.73597, 2.52119, 0.717606, 
										-1.38186, -16.1758, -15.1433, -10.8454, -2.73024, 5.19177, -9.13458, -6.22631, -16.3671, 18.563, 
										-14.1485, -19.3367, -15.2874, -13.6646, 7.35397, 14.258, -14.0546, 13.3238, -3.13919, -18.7306, 
										-10.3763, 5.35028, -6.48442, 12.3413, -10.8385, -9.06329, 2.68962, -15.1626, 2.20074, -14.7892, 
										5.55503, -19.1811, -10.965, 10.4117, -10.0266, 6.30479, -4.39648, 0.838862, -19.9215, -0.763596, 
										-0.598108, -14.07, -0.100311, 4.11447, -7.73464, -12.7463, -1.6275, -1.78927, -19.4226, 15.2333, 
										-0.519884, -9.79887, 0.583583, 12.9957, -17.4576, 9.74506, -16.0676, 5.23202, 14.5825, 6.13314, 
										10.4428, 0.13752, 6.95202, 19.4779, -9.45073, 16.9255, 5.78264, 6.15279, -2.23567, 5.86112, 
										-14.6108, 17.1662, 11.7911, 5.28889, 1.28069, -15.9435, 12.5425, 19.6532, 2.26721, 13.1199, 
										14.8865, -18.2527, -16.6789, -4.52991, 14.743, -14.1365, -14.7849, 18.6754, 11.0955, 19.7976, 
										4.80857, 1.53834, -0.0648417, -8.23941, 1.01619, 10.4844, -11.3139, -13.2012, -3.36278, 6.45039, 
										12.66, 2.02642, 3.61662, 4.45107, -12.6847, -15.1027, 8.50755, 19.8578, -15.4495, -9.22524, 
										12.9778, 19.437, -7.47791, 16.2989, -5.09291, -12.7349, -17.8376, 0.122241, -14.0595, 13.2579, 
										-0.0801206, 10.7491, -5.2038, 19.855, -17.4903, 15.8124, 10.3395, -8.80425, -17.3888, -13.0233, 
										17.6461, 15.2712, 9.0031, 1.26276, -0.277759, 16.3184, 6.16007, -11.7702, 16.1762, 10.7106, 
										-0.995453, 9.15404, 10.1476, 11.5266, 5.45291, -14.9453, 18.7917, 7.61526, 5.17691, -15.2677, 
										0.873122, -14.9032, 15.4814, 15.6693, -15.0482, 17.9911, 11.4817, 15.2913, -10.8132, 14.0929, 
										-17.732, -13.167, 9.3641, 11.2711, 8.09571, -10.9137, 7.58949, -5.74422, -2.68387, 3.76574, 
										-15.0336, 16.3207, -7.08022, 15.1139, 7.84731, 18.3727, -19.8314, 6.63905, 5.98795, 5.34551, 
										11.3713, -13.1389, 10.4423, 6.85269, -17.4696, 15.3941, 4.84375, 14.0121, 10.6854, 14.0306, 
										8.10502, 12.9534, -19.1365, -2.53088, 4.22448, 8.95923, 6.55545, -8.18603, -16.785, -16.1284, 
										15.5797, -11.8186, -19.8077, -11.5005, -16.7047, 8.03956, -13.1278, -16.5361, -5.32139, 12.8601, 
										8.8094, -13.9501, 19.7212, -0.748302, 12.9026, -17.7484, -5.35417, -2.25364, 16.2637, -14.6688, 
										-8.22308, 4.36871, 18.2846, -7.35957, -18.1622, 2.50913, -18.4003, 8.39328, 14.3231, -15.1853, 
										12.2649, 9.90282, -7.00397, 12.4571, 18.4023, -3.70869, 0.496673, -14.7255, -0.244798, 15.1753, 
										18.1346, -11.4354, -18.7748, 17.8558, 7.8163, 14.1278, -19.8926, -17.5379, -8.12582, 16.3711, 
										-12.2066, 3.65109, 0.739812, -13.922, 16.2915, 2.57764, 8.58715, 17.8912, -9.02909, 2.91026, 
										-17.2941, -16.7642, -7.18692, -4.29812, 15.6929, -8.78462, 11.9932, -3.81044, -3.51013, -8.2516, 
										-8.63516, -5.37552, 0.312999, -7.40995, -7.5197, -11.8707, -13.2821, -7.41228, -9.40858, -1.40796, 
										-11.0412, -1.61521, -17.7569, 9.69863};

			vector<float> data_out = {275.749, 49.6255, 221.241, 166.205, 0.172401, 11.0811, 11.3305, 229.727, 237.526, 35.0527, 93.5306,
										288.58, 292.358, 63.8734, 119.156, 327.244, 26.9116, 56.0169, 0.233526, 155.38, 51.3795, 79.7274, 
										226.684, 114.923, 92.5219, 106.886, 275.154, 122.126, 164.592, 1.45978, 258.805, 89.0203, 137.998, 
										352.943, 273.671, 75.1335, 20.6302, 46.5544, 182.553, 83.3601, 364.713, 10.1272, 147.441, 262.309, 
										77.891, 9.35363, 327.829, 35.8414, 242.547, 1.99008, 2.39651, 134.421, 93.3209, 11.5321, 365.812, 
										0.519685, 170.494, 246.929, 93.5054, 34.6531, 9.4729, 184.482, 21.8861, 26.78, 152.997, 66.1825, 
										37.9099, 148.178, 25.4184, 159.919, 1.69786, 17.1322, 111.016, 42.9608, 0.112262, 334.493, 272.379, 
										2.43015, 59.2533, 145.874, 290.035, 191.835, 380.881, 44.7652, 91.1156, 384.553, 196.847, 6.77447, 
										234.825, 13.6902, 132.492, 60.1215, 7.38508, 14.5253, 303.496, 107.088, 260.31, -61.5408, -37.5007, 
										-9.25142, -0.573767, -53.8465, 50.9735, 164.381, 42.0781, 30.738, -88.3417, 105.77, -279.853, 
										148.357, -154.442, 349.799, 79.3057, 102.272, 3.55377, 177.729, 100.743, -118.968, 47.2638, 
										-200.797, -99.8077, -55.3142, -107.562, 136.384, -139.051, -10.9504, 43.269, -143.06, -25.8527, 
										277.841, 91.8969, -166.261, -49.8035, -71.0402, -135.471, 57.5638, -83.9615, 2.66953, -241.898, 
										-12.3672, 5.27865, -43.0313, -1.81624, 24.6324, -120.459, 17.9813, 2.51948, 20.7448, -187.628, 
										51.7308, -9.94341, 7.06392, 7.62004, 204.214, 168.812, 57.3661, 49.453, -71.0635, -68.2207, 
										31.7386, -129.17, -1.11876, -42.8043, 237.101, -47.6474, -214.038, 7.5349, 25.4671, -23.556, 
										-38.4164, -4.89543, -313.956, 194.6, -8.24481, 9.8583, -192.563, 213.605, -272.206, -44.2473, 
										87.7814, 142.098, -357.935, 234.009, -11.7904, 225.922, 52.3056, 170.182, -144.806, 30.1526, 
										75.4529, 83.7707, 15.9192, -1.07674, -58.0428, -15.115, -135.166, -4.69769, -43.9445, 11.3194, 
										-97.767, -195.114, 11.9975, 34.9767, -75.6132, -216.889, -120.702, 92.867, -359.226, 80.1465, 
										69.0458, 6.27145, 242.285, 53.6013, -145.533, 76.679, -136.521, -171.577, -1.2638, -233.215, 
										146.514, -1.0279, 12.9872, -83.7157, 187.333, 205.463, -297.064, 171.046, -76.3149, -78.9807, 
										88.8591, 238.421, 139.428, 171.936, 4.01851, -3.37269, 264.292, -54.3662, -35.9977, 292.888, 
										64.1217, -15.5031, -12.9136, -15.7091, -133.64, 52.6766, -50.7528, 359.415, -5.48978, 67.5966, 
										-239.917, -8.44293, -87.7305, -47.6487, -212.827, 70.3993, 93.1026, -142.02, -124.399, 66.5779, 
										171.551, -89.3989, 166.509, 12.2016, 46.6523, 85.2999, 71.533, 2.5429, 105.057, -44.2944, 
										-5.87038, -115.723, 197.119, -120.579, -209.335, -153.149, 122.926, -189.3, 130.192, -84.0122, 
										13.9132, 174.254, 48.6144, -120.196, -53.1344, -47.4745, 58.6702, 84.3836, 145.002, 177.439, 
										98.8387, -120.556, -166.996, -7.9457, -8.4249, -14.2199, -135.793, -101.032, -48.4656, -162.33, 
										273.984, 266.39, -94.4556, -216.218, 208.043, 86.6581, -60.1716, -6.34395, -206.125, 38.1435, 
										-114.828, -132.635, -149.548, 189.695, 7.73637, 214.026, -196.139, -68.6906, -2.72288, 261.641, 
										-138.4, 96.5986, -82.074, 302.483, -63.7924, -82.4935, -17.12, -248.611, 76.6321, 273.535, 
										-48.3247, 148.926, 160.385, 61.8142, 38.0985, 333.193, -22.203, 7.73516, 20.7733, 0.378964, 
										-175.943, 175.185, -38.8335, -359.091, -12.8721, 102.06, 222.004, 192.358, -103.24, 25.0097, 
										-222.359, 57.1058, 18.8942, -9.15089, 113.259, -100.309, 31.3772, 43.2935, -226.25, -11.7651, 
										12.0459, -182.219, 109.88, -2.40802, 78.6089, 258.994, 13.6943, 92.319, -46.0219, -59.779, 
										114.288, 168.525, -35.9659, 2.98772, -145.309, 105.503, -30.8968, -203.536, 27.4257, 108.298, 
										10.9171, -30.005, -6.1559, -309.345, 100.365};


			float* data = &data_in[0];
			float* dl = &dt[0];

			dproduct(data, dl, nv, D, dn, dn_D);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_DELTA(data_in[c], data_out[c], 0.01);
		}

		void test_padding_dproductB( void )
		{
			unsigned long dn = 6;
			unsigned long D = 3;
			unsigned long nv = 1;
			unsigned long dnpg = 8;
			unsigned long dn_D = D * dnpg;

			vector<long> dt = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
			vector<float> data_in =   {6, 1, 7, 3, 3, 4};
			vector<float> data_out2 = {6, 1, 7, 3, 3, 4, 0, 0, 6, 1, 7, 3, 3, 4, 0, 0, 6, 1, 7, 3, 3, 4, 0, 0};
			vector<float> data_out1(dn_D * nv);

			float* data = &data_in[0];
			float* data_out = &data_out1[0];
			

			dproductB(data, dt, nv, dn, dnpg, dn_D, data_out);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_EQUALS(data_out1[c], data_out2[c]);
		}

		void test_dproductB( void )
		{
			unsigned long dn = 13;
			unsigned long D = 3;
			unsigned long nv = 1;
			unsigned long dnpg = 16;
			unsigned long dn_D = D * dnpg;

			
			vector<long> dt = {-1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1,
										-1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1};
			vector<float> data_in = {6, 4, 7, 5, 2, 7, 6, 3, 6, 3, 2, 4, 3};
			vector<float> data_out2 = {-6, -4, 7, 5, 2, 7, 6, -3, -6, -3, 2, 4, 3, 0, 0, 0, 6, 4, -7, -5, -2, -7, 6, -3, -6, -3, -2, 4,
										-3, 0, 0, 0, -6, -4, 7, 5, 2, 7, 6, 3, -6, 3, 2, 4, 3, 0, 0, 0};
			vector<float> data_out1(dn_D * nv);

			float* data = &data_in[0];
			float* data_out = &data_out1[0];
			
			dproductB(data, dt, nv, dn, dnpg, dn_D, data_out);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_EQUALS(data_out1[c], data_out2[c]);
		}

		void test_float_dproductB( void )
		{
			unsigned long dn = 5;
			unsigned long D = 2;
			unsigned long nv = 2;
			unsigned long dnpg = 8;
			unsigned long dn_D = D * dnpg;

			vector<long> dt = {1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1};
			vector<float> data_in = {-17.9795, 2.46397, -0.648622, 0.0787126, 10.6587, -3.19684, 10.2982, 1.46099, -12.6393, -4.09613};
			vector<float> data_out2 = {-17.9795, -2.46397, -0.648622, -0.0787126, -10.6587, 0, 0, 0, -17.9795, -2.46397, 0.648622, 
										0.0787126, 10.6587, 0, 0, 0, -3.19684, -10.2982, 1.46099, 12.6393, 4.09613, 0, 0, 0, -3.19684, 
										-10.2982, -1.46099, -12.6393, -4.09613, 0, 0, 0};			
			vector<float> data_out1(dn_D * nv);

			float* data = &data_in[0];
			float* data_out = &data_out1[0];
			
			dproductB(data, dt, nv, dn, dnpg, dn_D, data_out);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_EQUALS(data_out1[c], data_out2[c]);
		}

		void test_pn( void )
		{

			unsigned long D = 2;
			unsigned long nv = 1;
			unsigned long dnpg = 1UL << 3;
			unsigned long dn_D = D * dnpg;

			vector<unsigned long> pt = {6, 1, 0, 2, 5, 4, 3, 7, 
										2, 7, 3, 0, 1, 5, 4, 6};
			vector<float> data_in = {6, 0, 3, 8, 6, 2, 3, 1,
 										7, 1, 7, 5, 0, 5, 5, 7};
			vector<float> data_out = {3, 0, 6, 3, 2, 6, 8, 1,
										7, 7, 5, 7, 1, 5, 0, 5};

			float* data = &data_in[0];
			unsigned long* p = &pt[0];
			
			pn(data, p, nv, D, dnpg, dn_D);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);
		}

		void test_float_pn( void )
		{

			unsigned long D = 2;
			unsigned long nv = 3;
			unsigned long dnpg = 1UL << 3;
			unsigned long dn_D = D * dnpg;

			vector<unsigned long> pt = {2, 6, 0, 7, 4, 3, 5, 1,
 			  									5, 4, 0, 2, 6, 3, 1, 7};

			vector<float> data_in = {-0.923757, -1.73513, -5.73986, -5.78223, 9.7221, 14.2887, -6.69936, 0.0702897,
 										-3.03486, 8.32579, 16.199, -6.98968, -1.50254, 4.91236, -6.36348, 14.1976,
		 							-7.07566, 10.7438, 3.95253, -9.04085, 2.2813, -12.8007, 10.1868, -17.5466,
 										13.3548, -16.8699, -1.04428, 8.68342, -1.00041, -1.28772, -4.56084, 18.0758,
									16.9772, 9.6993, -7.70639, 6.69925, 3.98803, 5.59425, -13.2305, -19.0468,
 										-6.07996, -17.0314, -6.0365, 12.4175, 7.88091, 7.60002, 6.61505, -19.1947};

			vector<float> data_out = {-5.73986, -6.69936, -0.923757, 0.0702897, 9.7221, -5.78223, 14.2887, -1.73513,
 											4.91236, -1.50254, -3.03486, 16.199, -6.36348, -6.98968, 8.32579, 14.1976,
 										3.95253, 10.1868, -7.07566, -17.5466, 2.2813, -9.04085, -12.8007, 10.7438,
											-1.28772, -1.00041, 13.3548, -1.04428, -4.56084, 8.68342, -16.8699, 18.0758,
		 								-7.70639, -13.2305, 16.9772, -19.0468, 3.98803, 6.69925, 5.59425, 9.6993,
											7.60002, 7.88091, -6.07996, -6.0365, 6.61505, 12.4175, -17.0314, -19.1947};

			float* data = &data_in[0];
			unsigned long* p = &pt[0];
			
			pn(data, p, nv, D, dnpg, dn_D);

			for(unsigned long c = 0; c < dn_D * nv; ++c)
				TS_ASSERT_EQUALS(data_in[c], data_out[c]);
		}		
};
