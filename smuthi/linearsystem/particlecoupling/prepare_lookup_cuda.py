"""This module contains CUDA source code for the preparation of coupling 
matrix lookups."""


# This cuda kernel is used for the calculation of volume lookup tables.
volume_lookup_assembly_code = """
    #define BLOCKSIZE %i
    #define RHO_ARRAY_LENGTH %i
    #define Z_ARRAY_LENGTH %i
    #define K_ARRAY_LENGTH %i
    
    __global__ void helper(const float *re_bes_jac, const float *im_bes_jac, const float *re_belbee, 
                            const float *im_belbee, const float *re_d_kappa, const float *im_d_kappa, 
                            float *re_result, float  *im_result)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= RHO_ARRAY_LENGTH * Z_ARRAY_LENGTH) return;
        
        unsigned int i_rho = i / Z_ARRAY_LENGTH;
        unsigned int i_z = i %% Z_ARRAY_LENGTH; 
        
        float re_res = 0.0;
        float im_res = 0.0;
        
        int i_kr = i_rho * K_ARRAY_LENGTH;
        int i_kz = i_z * K_ARRAY_LENGTH;

        float re_integrand_kp1 = re_bes_jac[i_kr] * re_belbee[i_kz] - im_bes_jac[i_kr] * im_belbee[i_kz];
        float im_integrand_kp1 = re_bes_jac[i_kr] * im_belbee[i_kz] + im_bes_jac[i_kr] * re_belbee[i_kz];

        for (int i_k=0; i_k<(K_ARRAY_LENGTH-1); i_k++)
        {
            i_kr = i_rho * K_ARRAY_LENGTH + i_k;
            i_kz = i_z * K_ARRAY_LENGTH + i_k;

            float re_integrand = re_integrand_kp1;
            float im_integrand = im_integrand_kp1;

            re_integrand_kp1 = re_bes_jac[i_kr+1] * re_belbee[i_kz+1] - im_bes_jac[i_kr+1] * im_belbee[i_kz+1];
            im_integrand_kp1 = re_bes_jac[i_kr+1] * im_belbee[i_kz+1] + im_bes_jac[i_kr+1] * re_belbee[i_kz+1];
            
            float re_sint = re_integrand + re_integrand_kp1;
            float im_sint = im_integrand + im_integrand_kp1;
            
            re_res += 0.5 * (re_sint * re_d_kappa[i_k] - im_sint * im_d_kappa[i_k]);
            im_res += 0.5 * (re_sint * im_d_kappa[i_k] + im_sint * re_d_kappa[i_k]);
        }
        
        re_result[i] = re_res;
        im_result[i] = im_res;
        
    }"""

# This cuda kernel is used for the calculation of radial lookup tables.
radial_lookup_assembly_code = """
    #define BLOCKSIZE %i
    #define RHO_ARRAY_LENGTH %i
    #define K_ARRAY_LENGTH %i
    
    __global__ void helper(const float *re_bes_jac, const float *im_bes_jac, const float *re_belbee, 
                            const float *im_belbee, const float *re_d_kappa, const float *im_d_kappa, 
                            float *re_result, float  *im_result)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= RHO_ARRAY_LENGTH) return;
        
        float re_res = 0.0;
        float im_res = 0.0;
        
        int i_kr = i * K_ARRAY_LENGTH;

        float re_integrand_kp1 = re_bes_jac[i_kr] * re_belbee[0] - im_bes_jac[i_kr] * im_belbee[0];
        float im_integrand_kp1 = re_bes_jac[i_kr] * im_belbee[0] + im_bes_jac[i_kr] * re_belbee[0];

        for (int i_k=0; i_k<(K_ARRAY_LENGTH-1); i_k++)
        {
            i_kr = i * K_ARRAY_LENGTH + i_k;
            
            float re_integrand = re_integrand_kp1;
            float im_integrand = im_integrand_kp1;

            re_integrand_kp1 = re_bes_jac[i_kr+1] * re_belbee[i_k+1] - im_bes_jac[i_kr+1] * im_belbee[i_k+1];
            im_integrand_kp1 = re_bes_jac[i_kr+1] * im_belbee[i_k+1] + im_bes_jac[i_kr+1] * re_belbee[i_k+1];
            
            float re_sint = re_integrand + re_integrand_kp1;
            float im_sint = im_integrand + im_integrand_kp1;
            
            re_res += 0.5 * (re_sint * re_d_kappa[i_k] - im_sint * im_d_kappa[i_k]);
            im_res += 0.5 * (re_sint * im_d_kappa[i_k] + im_sint * re_d_kappa[i_k]);
        }
        
        re_result[i] = re_res;
        im_result[i] = im_res;
        
    }"""