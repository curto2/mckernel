/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}@estudiants.urv.cat 						    */

#include "../hpp/Factory_McKernel.hpp"
#include "../hpp/sse.h"

//McKernel  
McKernel::McKernel( float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const float sigma)
:M_data(data), M_nv(nv), M_dn(dn)
{
	random_device rd;
	mt19937 gr(rd());
		
	//Compute padding dimension 
	if((M_dn & (M_dn - 1)) != 0) //Test if power of 2
	{	           
  		unsigned long p2 = ceil(log2(M_dn));
  		M_dnpg = 1UL << p2;
	}else
		M_dnpg = M_dn;

	M_D = D / M_dn;	
	M_dn_D = M_dnpg * M_D; 

	//Initializing G,C,Pi,B
	dl_G.resize(M_dn_D);
	dl_Pi.resize(M_dn_D);
	dl_C.resize(M_dn_D);
	dl_B.resize(M_dn_D);

	long* dlv_B = &dl_B[0];
        unsigned long* p = &dl_Pi[0];

        M_data_out = new float[M_dn_D * M_nv];
	M_features = new float[2 * M_dn_D * M_nv];

	//Generate B from uniform(0,1)
	uniform_real_distribution<> urd(0, 1);
	for (unsigned long c = 0; c < M_dn_D; ++c)
		if(urd(gr) < 0.5)
			dlv_B[c] = -1; 
		else
			dlv_B[c] = 1;

	//Generate Pi using Reservoir Sampling
	for (unsigned long z = 0; z < M_dn_D; ++z)
	   p[z] = z % M_dnpg;

	for(unsigned long c = 0; c < M_D; ++c)
		fy(p + c * M_dnpg, M_dnpg);

	dlv_B = NULL;
 	p = NULL;
}

//McFeatures
void McKernel::McFeatures()
{
  	unsigned long* p = &dl_Pi[0]; 
  	float* dlv_G = &dl_G[0];
  	float* dlv_C = &dl_C[0];

	//Computing McKernel: 1/(sigma * sqrt(d)) * C * H * G * Pi * HB * x

  	//Bx and padding
  	dproductB(M_data, dl_B, M_nv, M_dn, M_dnpg, M_dn_D, M_data_out);

  	//HBx
  	for (unsigned long c = 0; c < M_nv; ++c)
  		for(unsigned long z = 0; z < M_D; ++z)
    			fwh(M_data_out + c * M_dn_D + z * M_dnpg, log2(M_dnpg));

  	//PiHBx
  	pn(M_data_out, p, M_nv, M_D, M_dnpg, M_dn_D);

  	//GPiHBx
        dproduct(M_data_out, dlv_G, M_nv, M_D, M_dnpg, M_dn_D);

  	//HGPiHBx
    	for (unsigned long c = 0; c < M_nv; ++c)
  		for(unsigned long z = 0; z < M_D; ++z)
    			fwh(M_data_out + c * M_dn_D + z * M_dnpg, log2(M_dnpg));

  	//CHGPiHBx
        dproduct(M_data_out, dlv_C, M_nv, M_D, M_dnpg, M_dn_D);
  
  	dlv_G = NULL;
  	dlv_C = NULL;
  	p = NULL;
}

//McEvaluate
void McKernel::McEvaluate()
{
	const float c = 1.0;
	//const float c = 1.0 / sqrt( M_dn_D );
	const __m128 vc = _mm_set1_ps(c);
	
	for (unsigned long c = 0; c < M_nv; ++c){

        unsigned long z;
        for(z = 0; z <= M_dn_D - 4; z += 4)
        {

    	 	unsigned long index = z + c * M_dn_D;
    		unsigned long index_features = z + 2 * c * M_dn_D;

    		const __m128 src0 = _mm_loadu_ps((float *)&M_data_out[index]);
			__m128 a0, a1;

		//Complex mapping [c * sin(Zx) , c * cos(Zx)]
		sincos_ps(src0, &a0, &a1);	
		const __m128 v0 = _mm_mul_ps(vc, a0);
		const __m128 v1 = _mm_mul_ps(vc, a1);

                _mm_storeu_ps((float *)&M_features[index_features], v0);
                _mm_storeu_ps((float *)&M_features[index_features + M_dn_D], v1);

        }

       	for(unsigned long k = z; k < M_dn_D; ++k )
       	{
           unsigned long index = k + c * M_dn_D;
           unsigned long index_features = k + 2 * c * M_dn_D;

           M_features[index_features] = c * cos(M_data_out[index]);
           M_features[index_features + M_dn_D] = c * sin(M_data_out[index]);
       	}

   }

}

//RBF GAUSSIAN 
RBF_GAUSSIAN::RBF_GAUSSIAN(float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const float sigma)
:McKernel(data, nv, dn, D, sigma)
{
    	random_device rd;
    	mt19937 gr(rd());

    	vector<float> scalar_C(M_D);
    
    	float* dlv_G = &dl_G[0];
	float* dlv_C = &dl_C[0];
	float* sv_C = &scalar_C[0];

	//Generate G using N(0,1) and compute FROBENIUS Norm
	normal_distribution<> nd(0, 1);
	fill(scalar_C.begin(), scalar_C.end(), 0.0);
	for (unsigned long c = 0; c < M_dn_D; ++c)
	{
		dlv_G[c] = nd(gr);
		sv_C[c / M_dnpg] += dlv_G[c] * dlv_G[c]; 
	}

	for(unsigned long c = 0; c < M_D; ++c)
		sv_C[c] = 1.0 / sqrt(sv_C[c]);

	//Generate C using Chi Distribution
	float sigma_factor = 1.0 / (sigma * sqrt(M_dnpg));
    	chi_squared_distribution<> csd(M_dnpg); 
    	for (unsigned long k = 0; k < M_D; ++k)
		for (unsigned long c = 0; c < M_dnpg; ++c)
	 	       dlv_C[c + k * M_dnpg] = sigma_factor * sv_C[k] * sqrt(csd(gr));

	dlv_G = NULL;
    	dlv_C = NULL;
    	sv_C = NULL;
}

//RBF MATÉRN
RBF_MATERN::RBF_MATERN(float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, 
	const float sigma, const unsigned long t):McKernel(data, nv, dn, D, sigma)
{
	random_device rd;
    	mt19937 gr(rd());

    	vector<float> scalar_C(M_D);

   	float* dlv_G = &dl_G[0];
	float* dlv_C = &dl_C[0];
	float* sv_C = &scalar_C[0];	

	uniform_real_distribution<> urd(0, 1);
	normal_distribution<> nd(0, 1);

	//Generate G using N(0,1) and compute FROBENIUS Norm 
	fill(scalar_C.begin(), scalar_C.end(), 0.0);
	for (unsigned long c = 0; c < M_dn_D; ++c)
	{
		dlv_G[c] = nd(gr);
		sv_C[c / M_dnpg] += dlv_G[c] * dlv_G[c]; 
	}

	for(unsigned long c = 0; c < M_D; ++c)
		sv_C[c] = 1.0 / sqrt(sv_C[c]);

	//Generate C for MATÉRN Kernel
	float sigma_factor = 1.0 / (sigma * sqrt(M_dnpg));
    	for (unsigned long k = 0; k < M_D; ++k)
	{
		for (unsigned long c = 0; c < M_dnpg; ++c)
		{
    			float norm_psi;

	    		//Draw t iid samples psi_c uniformly from Sd
			vector<float> psi_n(M_dnpg);
		    	for (unsigned long n = 0; n < t; ++n)
			{

				vector<float> psi(M_dnpg);
				norm_psi = 0.0;

			        //Generate sample z from n-dimensional standard normal distribution
			        for(unsigned long r = 0; r < M_dnpg; ++r)
			        {
			    		psi[r] = nd(gr);
			    		norm_psi += psi[r] * psi[r];
			    	}

			        //Compute ||z||
			        norm_psi = sqrt(norm_psi);
			    
			        //Generate sample from uniform distribution U^1/n
			        float u = pow(urd(gr), (1.0/M_dn_D)); 

			        //Return r * U^1/n * z/||z||
			        for(unsigned long m = 0; m < M_dnpg; ++m)
			    		psi_n[m] += psi[m] * u / norm_psi;			    

			}

			//Compute L2 norm of psi
			norm_psi = 0.0;
			for(unsigned long r = 0; r < M_dnpg; ++r)
			    norm_psi += psi_n[r] * psi_n[r];
			
			//Assign to C_cc   
			dlv_C[c + k * M_dnpg] = sigma_factor * sv_C[k] * sqrt(norm_psi); 
		}
	}

	dlv_G = NULL;
        dlv_C = NULL;
        sv_C = NULL;
}

//Factory McKernel
McKernel* FactoryMcKernel::createMcKernel(TypeMcKernel typemckernel, float* data, const unsigned long nv, 
	const unsigned long dn, const unsigned long D, const float sigma, const unsigned long t)
{
	switch(typemckernel)
	{
		case RBF:
			return new RBF_GAUSSIAN(data, nv, dn, D, sigma);
		case MRBF:
			return new RBF_MATERN(data, nv, dn, D, sigma, t);
		//You can include a new one here
	}
	throw "Invalid Type McKernel.";
}
