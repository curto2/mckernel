/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time 
   Curtó, Zarza, Yang, Smola, De La Torre, Ngo, and Van Gool 		    

   Authors: Curtó and Zarza
   {curto,zarza}@tinet.cat 						    */

#include "../hpp/Factory_McKernel.hpp"
#include "../hpp/sse.h"

//McKernel  
McKernel::McKernel( float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma)
:M_data(data), M_nv(nv), M_dn(dn)
{
	//Generate seeds
	unsigned int u1_seed = MurmurHash2(&seed, sizeof(seed), 0x1337beef);
	unsigned int u2_seed = MurmurHash2(&seed, sizeof(seed), 0x1337face);

	//PRNG Declaration
	U_PNG u1_png;
	U_PNG u2_png;

	//PRNG Seed
  	u1_png.SetState(u1_seed);
    	u2_png.SetState(u2_seed);
		
	//Compute padding dimension 
	if((M_dn & (M_dn - 1)) != 0) //Test if power of 2
	{	           
  		unsigned long p2 = ceil(log2(M_dn));
  		M_dnpg = 1UL << p2;
	}else
		M_dnpg = M_dn;

	M_D = D / M_dn;
	M_dn_D = M_dnpg * M_D;	

	//Initializing G,S,Pi,B
	dl_G.resize(M_dn_D);
	dl_Pi.resize(M_dn_D);
	dl_S.resize(M_dn_D);
	dl_B.resize(M_dn_D);

	long* dlv_B = &dl_B[0];
        unsigned long* p = &dl_Pi[0];

        M_data_out = new float[M_dn_D * M_nv];
	M_features = new float[2 * M_dn_D * M_nv];

	//Generate B from uniform(0,1)
	for (unsigned long i = 0; i < M_dn_D; ++i)
		if(u1_png.GetUniform(i) < 0.5)
			dlv_B[i] = -1; 
		else
			dlv_B[i] = 1;

	//Generate Pi using Reservoir Sampling
	for (unsigned long j = 0; j < M_dn_D; ++j)
	   p[j] = j % M_dnpg;

	for(unsigned long i = 0; i < M_D; ++i)
		fys(p + i * M_dnpg, M_dnpg, u2_png);

	dlv_B = NULL;
 	p = NULL;
}

//McFeatures
void McKernel::McFeatures()
{
  	unsigned long* p = &dl_Pi[0]; 
  	float* dlv_G = &dl_G[0];
  	float* dlv_S = &dl_S[0];

	//Computing McKernel: 1/(sigma * sqrt(d)) * S * H * G * Pi * HB * x

  	//Bx and padding
  	dproductB(M_data, dl_B, M_nv, M_dn, M_dnpg, M_dn_D, M_data_out);

  	//HBx
  	for (unsigned long i = 0; i < M_nv; ++i)
  		for(unsigned long j = 0; j < M_D; ++j)
    			fwht(M_data_out + i * M_dn_D + j * M_dnpg, log2(M_dnpg));

  	//PiHBx
  	pn(M_data_out, p, M_nv, M_D, M_dnpg, M_dn_D);

  	//GPiHBx
        dproduct(M_data_out, dlv_G, M_nv, M_D, M_dnpg, M_dn_D);

  	//HGPiHBx
    	for (unsigned long i = 0; i < M_nv; ++i)
  		for(unsigned long j = 0; j < M_D; ++j)
    			fwht(M_data_out + i * M_dn_D + j * M_dnpg, log2(M_dnpg));

  	//SHGPiHBx
        dproduct(M_data_out, dlv_S, M_nv, M_D, M_dnpg, M_dn_D);
  
  	dlv_G = NULL;
  	dlv_S = NULL;
  	p = NULL;
}

//McEvaluate
void McKernel::McEvaluate()
{
	const float c = 1.0;
	//const float c = 1.0 / sqrt( M_dn_D );
	const __m128 vc = _mm_set1_ps(c);
	
	for (unsigned long i = 0; i < M_nv; ++i){

        unsigned long j;
        for(j = 0; j <= M_dn_D - 4; j += 4)
        {

    	 	unsigned long index = j + i * M_dn_D;
    		unsigned long index_features = j + 2 * i * M_dn_D;

    		const __m128 src0 = _mm_loadu_ps((float *)&M_data_out[index]);
			__m128 a0, a1;

		//Complex mapping [c * sin(data) , c * cos(data)]
		sincos_ps(src0, &a0, &a1);	
		const __m128 v0 = _mm_mul_ps(vc, a0);
		const __m128 v1 = _mm_mul_ps(vc, a1);

                _mm_storeu_ps((float *)&M_features[index_features], v0);
                _mm_storeu_ps((float *)&M_features[index_features + M_dn_D], v1);

        }

       	for(unsigned long k = j; k < M_dn_D; ++k )
       	{
           unsigned long index = k + i * M_dn_D;
           unsigned long index_features = k + 2 * i * M_dn_D;

           M_features[index_features] = c * cos(M_data_out[index]);
           M_features[index_features + M_dn_D] = c * sin(M_data_out[index]);
       	}

   }

}

//RBF Gaussian 
RBF_Gaussian::RBF_Gaussian(float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma)
:McKernel(data, nv, dn, D, seed, sigma)
{
	//Generate seeds
	unsigned int n1_seed = MurmurHash2(&seed, sizeof(seed), 0x1337feed);
	unsigned int n2_seed = MurmurHash2(&seed, sizeof(seed), 0x1337df2f);
	unsigned int c1_seed = MurmurHash2(&seed, sizeof(seed), 0x1337abcd);
	unsigned int c2_seed = MurmurHash2(&seed, sizeof(seed), 0x1337eadf);

	//PRNG Declaration
	NC_PNG n_png;
	NC_PNG c_png;

	//PRNG Seed
        n_png.SetState(n1_seed, n2_seed);
        c_png.SetState(c1_seed, c2_seed);

    	vector<float> scalar_S(M_D);
    
    	float* dlv_G = &dl_G[0];
	float* dlv_S = &dl_S[0];
	float* sv_S = &scalar_S[0];

	//Generate G using N(0,1) and compute Frobenius Norm
	fill(scalar_S.begin(), scalar_S.end(), 0.0);
	for (unsigned long i = 0; i < M_dn_D; ++i)
	{
		dlv_G[i] = n_png.GetNormal(i);
		sv_S[i / M_dnpg] += dlv_G[i] * dlv_G[i]; 
	}

	for(unsigned long i = 0; i < M_D; ++i)
		sv_S[i] = 1.0 / sqrt(sv_S[i]);

	//Generate S using Chi Distribution
	float sigma_factor = 1.0 / (sigma * sqrt(M_dnpg)); 
    	for (unsigned long k = 0; k < M_D; ++k)
		for (unsigned long i = 0; i < M_dnpg; ++i)
	 	       dlv_S[i + k * M_dnpg] = sigma_factor * sv_S[k] * sqrt(c_png.GetChiSquared(i + k * M_dnpg, M_dnpg));

	dlv_G = NULL;
    	dlv_S = NULL;
    	sv_S = NULL;
}

//RBF Matern
RBF_Matern::RBF_Matern(float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const unsigned long seed, 
	const float sigma, const unsigned long t):McKernel(data, nv, dn, D, seed, sigma)
{
	//Generate seeds
	unsigned int n1_seed = MurmurHash2(&seed, sizeof(seed), 0x1337feed);
	unsigned int n2_seed = MurmurHash2(&seed, sizeof(seed), 0x1337df2f);
	unsigned int n3_seed = MurmurHash2(&seed, sizeof(seed), 0x1337abcd);
	unsigned int n4_seed = MurmurHash2(&seed, sizeof(seed), 0x1337eadf);
	unsigned int u3_seed = MurmurHash2(&seed, sizeof(seed), 0x1337baca);

	//PRNG Declaration
	NC_PNG n1_png;
	NC_PNG n2_png;
	U_PNG u3_png;

	//PRNG Seed
    	n1_png.SetState(n1_seed, n2_seed);
    	n2_png.SetState(n3_seed, n4_seed);
    	u3_png.SetState(u3_seed);

    	vector<float> scalar_S(M_D);

   	float* dlv_G = &dl_G[0];
	float* dlv_S = &dl_S[0];
	float* sv_S = &scalar_S[0];	

	//Generate G using N(0,1) and compute Frobenius Norm 
	fill(scalar_S.begin(), scalar_S.end(), 0.0);
	for (unsigned long i = 0; i < M_dn_D; ++i)
	{
		dlv_G[i] = n1_png.GetNormal(i);
		sv_S[i / M_dnpg] += dlv_G[i] * dlv_G[i]; 
	}

	for(unsigned long i = 0; i < M_D; ++i)
		sv_S[i] = 1.0 / sqrt(sv_S[i]);

	//Generate S for Matern Kernel
	float sigma_factor = 1.0 / (sigma * sqrt(M_dnpg));
    	for (unsigned long k = 0; k < M_D; ++k)
	{
		for (unsigned long i = 0; i < M_dnpg; ++i)
		{
    			float norm_psi;
			unsigned long ix = k * M_dnpg + i;
	    		//Draw t iid samples psi_i uniformly from Sd
			vector<float> psi_n(M_dnpg);
		    	for (unsigned long n = 0; n < t; ++n)
			{

				vector<float> psi(M_dnpg);
				norm_psi = 0.0;

			        //Generate sample z from n-dimensional standard normal distribution
			        for(unsigned long r = 0; r < M_dnpg; ++r)
			        {
			    		psi[r] = n2_png.GetNormal(r + n * M_dnpg + ix);
			    		norm_psi += psi[r] * psi[r];
			    	}

			        //Compute ||z||
			        norm_psi = sqrt(norm_psi);
			    
			        //Generate sample from uniform distribution U^1/n
			        float u = pow(u3_png.GetUniform(n + ix), (1.0/M_dn_D)); 

			        //Return r * U^1/n * z/||z||
			        for(unsigned long m = 0; m < M_dnpg; ++m)
			    		psi_n[m] += psi[m] * u / norm_psi;			    

			}

			//Compute L2 norm of psi
			norm_psi = 0.0;
			for(unsigned long r = 0; r < M_dnpg; ++r)
			    norm_psi += psi_n[r] * psi_n[r];
			
			//Assign to S_ii   
			dlv_S[ix] = sigma_factor * sv_S[k] * sqrt(norm_psi); 
		}
	}

	dlv_G = NULL;
        dlv_S = NULL;
        sv_S = NULL;
}

//Factory McKernel
McKernel* FactoryMcKernel::createMcKernel(TypeMcKernel typemckernel, float* data, const unsigned long nv, 
	const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma, const unsigned long t)
{
	switch(typemckernel)
	{
		case RBF:
			return new RBF_Gaussian(data, nv, dn, D, seed, sigma);
		case MRBF:
			return new RBF_Matern(data, nv, dn, D, seed, sigma, t);
		//You can include a new one here
	}
	throw "Invalid Type McKernel.";
}
