"""This module contains CUDA source code for the evaluation of the electric 
field from a VWF expansion."""

# This cuda kernel is used for the evaluation of the electric field of plane wave expansions.
pwe_electric_field_evaluation_code = """
    #define LEN_X %i
    #define LEN_K %i
    #define LEN_A %i
    #define RE_ONE_OVER_K %.10f
    #define IM_ONE_OVER_K %.10f
    
    __device__ void alpha_integral(const int i_k, const float x, const float y, const float z, 
                                   const float re_kp, const float im_kp, const float re_kz, const float im_kz, 
                                   const float *alpha_array, 
                                   const float *re_g_te_array, const float *im_g_te_array,
                                   const float *re_g_tm_array, const float *im_g_tm_array,
                                   float *re_integral_x, float *im_integral_x,
                                   float *re_integral_y, float *im_integral_y,
                                   float *re_integral_z, float *im_integral_z)
    {
        float a = 0.0;
        
        float re_x_result = 0.0;
        float im_x_result = 0.0;
        float re_y_result = 0.0;
        float im_y_result = 0.0;
        float re_z_result = 0.0;
        float im_z_result = 0.0;
        
        float re_e_x_alpha_integrand = 0.0;
        float im_e_x_alpha_integrand = 0.0;
        float re_e_y_alpha_integrand = 0.0;
        float im_e_y_alpha_integrand = 0.0;
        float re_e_z_alpha_integrand = 0.0;
        float im_e_z_alpha_integrand = 0.0;

        for (int i_a=0; i_a<LEN_A; i_a++)
        {
            float a_old = a;
            a = alpha_array[i_a];
            float d_a = a - a_old;
            float sin_a = sinf(a);
            float cos_a = cosf(a);
            
            float re_ikr = -im_kp * (cos_a * x + sin_a * y) - im_kz * z;
            float im_ikr = re_kp * (cos_a * x + sin_a * y) + re_kz * z;
            
            float ereikr = expf(re_ikr);
            float re_eikr =  ereikr * cosf(im_ikr);
            float im_eikr =  ereikr * sinf(im_ikr);
            
            // integrand
            float re_e_x_alpha_integrand_old = re_e_x_alpha_integrand;
            float im_e_x_alpha_integrand_old = im_e_x_alpha_integrand;
            float re_e_y_alpha_integrand_old = re_e_y_alpha_integrand;
            float im_e_y_alpha_integrand_old = im_e_y_alpha_integrand;
            float re_e_z_alpha_integrand_old = re_e_z_alpha_integrand;
            float im_e_z_alpha_integrand_old = im_e_z_alpha_integrand;
            
            int i_ka = i_k * LEN_A + i_a;
            
            // pol=TE
            float re_g = re_g_te_array[i_ka];
            float im_g = im_g_te_array[i_ka];

            float re_geikr = re_g * re_eikr - im_g * im_eikr;
            float im_geikr = re_g * im_eikr + im_g * re_eikr;
            
            re_e_x_alpha_integrand = -sin_a * re_geikr;
            im_e_x_alpha_integrand = -sin_a * im_geikr;
            re_e_y_alpha_integrand = cos_a * re_geikr;
            im_e_y_alpha_integrand = cos_a * im_geikr;

            // pol=TM
            re_g = re_g_tm_array[i_ka];
            im_g = im_g_tm_array[i_ka];

            re_geikr = re_g * re_eikr - im_g * im_eikr;
            im_geikr = re_g * im_eikr + im_g * re_eikr;
            
            float re_kzgeikr = re_kz * re_geikr - im_kz * im_geikr;
            float im_kzgeikr = re_kz * im_geikr + im_kz * re_geikr;
            
            float re_kzkgeikr = re_kzgeikr * RE_ONE_OVER_K - im_kzgeikr * IM_ONE_OVER_K;
            float im_kzkgeikr = re_kzgeikr * IM_ONE_OVER_K + im_kzgeikr * RE_ONE_OVER_K;
            
            re_e_x_alpha_integrand += cos_a * re_kzkgeikr;
            im_e_x_alpha_integrand += cos_a * im_kzkgeikr;
            re_e_y_alpha_integrand += sin_a * re_kzkgeikr;
            im_e_y_alpha_integrand += sin_a * im_kzkgeikr;

            float re_kpgeikr = re_kp * re_geikr - im_kp * im_geikr;
            float im_kpgeikr = re_kp * im_geikr + im_kp * re_geikr;
            re_e_z_alpha_integrand = -(re_kpgeikr * RE_ONE_OVER_K - im_kpgeikr * IM_ONE_OVER_K);
            im_e_z_alpha_integrand = -(re_kpgeikr * IM_ONE_OVER_K + im_kpgeikr * RE_ONE_OVER_K);
            
            if (i_a>0)
            {
                re_x_result += 0.5 * d_a * (re_e_x_alpha_integrand + re_e_x_alpha_integrand_old);
                im_x_result += 0.5 * d_a * (im_e_x_alpha_integrand + im_e_x_alpha_integrand_old);
                re_y_result += 0.5 * d_a * (re_e_y_alpha_integrand + re_e_y_alpha_integrand_old);
                im_y_result += 0.5 * d_a * (im_e_y_alpha_integrand + im_e_y_alpha_integrand_old);
                re_z_result += 0.5 * d_a * (re_e_z_alpha_integrand + re_e_z_alpha_integrand_old);
                im_z_result += 0.5 * d_a * (im_e_z_alpha_integrand + im_e_z_alpha_integrand_old);
            }
        }
        re_integral_x[0] = re_x_result;
        im_integral_x[0] = im_x_result;
        re_integral_y[0] = re_y_result;
        im_integral_y[0] = im_y_result;
        re_integral_z[0] = re_z_result;
        im_integral_z[0] = im_z_result;
    }
    
    
    __global__ void electric_field(const float *re_kp_array, const float *im_kp_array, 
                                   const float *re_kz_array, const float *im_kz_array,
                                   const float *alpha_array,  
                                   const float *x_array, const float *y_array, const float *z_array,
                                   const float *re_g_te_array, const float *im_g_te_array,
                                   const float *re_g_tm_array, const float *im_g_tm_array,
                                   float *re_e_x, float *im_e_x, float *re_e_y, float *im_e_y, 
                                   float *re_e_z, float *im_e_z)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= LEN_X) return;
        
        float x = x_array[i];
        float y = y_array[i];
        float z = z_array[i];
        
        float re_kp = 0.0;
        float im_kp = 0.0;

        float re_x_result = 0.0;
        float im_x_result = 0.0;
        float re_y_result = 0.0;
        float im_y_result = 0.0;
        float re_z_result = 0.0;
        float im_z_result = 0.0;
        
        float re_x_kappa_integrand = 0.0;
        float im_x_kappa_integrand = 0.0;
        float re_y_kappa_integrand = 0.0;
        float im_y_kappa_integrand = 0.0;
        float re_z_kappa_integrand = 0.0;
        float im_z_kappa_integrand = 0.0;
        
        for (int i_k=0; i_k<LEN_K; i_k++)
        {
            float re_kp_old = re_kp;
            float im_kp_old = im_kp;
            
            re_kp = re_kp_array[i_k];
            im_kp = im_kp_array[i_k];
            
            float re_d_kp = re_kp - re_kp_old;
            float im_d_kp = im_kp - im_kp_old;

            float re_kz = re_kz_array[i_k];
            float im_kz = im_kz_array[i_k];
            
            float re_x_kappa_integrand_old = re_x_kappa_integrand;
            float im_x_kappa_integrand_old = im_x_kappa_integrand;
            float re_y_kappa_integrand_old = re_y_kappa_integrand;
            float im_y_kappa_integrand_old = im_y_kappa_integrand;
            float re_z_kappa_integrand_old = re_z_kappa_integrand;
            float im_z_kappa_integrand_old = im_z_kappa_integrand;
            
            float re_x_alpha_itgrl = 0.0;
            float im_x_alpha_itgrl = 0.0;
            float re_y_alpha_itgrl = 0.0;
            float im_y_alpha_itgrl = 0.0;
            float re_z_alpha_itgrl = 0.0;
            float im_z_alpha_itgrl = 0.0;
            
            alpha_integral(i_k, x, y, z, re_kp, im_kp, re_kz, im_kz, alpha_array, 
                           re_g_te_array, im_g_te_array, re_g_tm_array, im_g_tm_array,
                           &re_x_alpha_itgrl, &im_x_alpha_itgrl, &re_y_alpha_itgrl, &im_y_alpha_itgrl, 
                           &re_z_alpha_itgrl, &im_z_alpha_itgrl);
                           
            re_x_kappa_integrand = re_x_alpha_itgrl * re_kp - im_x_alpha_itgrl * im_kp;
            im_x_kappa_integrand = re_x_alpha_itgrl * im_kp + im_x_alpha_itgrl * re_kp;
            re_y_kappa_integrand = re_y_alpha_itgrl * re_kp - im_y_alpha_itgrl * im_kp;
            im_y_kappa_integrand = re_y_alpha_itgrl * im_kp + im_y_alpha_itgrl * re_kp;
            re_z_kappa_integrand = re_z_alpha_itgrl * re_kp - im_z_alpha_itgrl * im_kp;
            im_z_kappa_integrand = re_z_alpha_itgrl * im_kp + im_z_alpha_itgrl * re_kp;
            
            if (i_k>0)
            {
                re_x_result += 0.5 * (re_d_kp * (re_x_kappa_integrand + re_x_kappa_integrand_old) 
                                      - im_d_kp * (im_x_kappa_integrand + im_x_kappa_integrand_old));
                im_x_result += 0.5 * (re_d_kp * (im_x_kappa_integrand + im_x_kappa_integrand_old) 
                                      + im_d_kp * (re_x_kappa_integrand + re_x_kappa_integrand_old));
                re_y_result += 0.5 * (re_d_kp * (re_y_kappa_integrand + re_y_kappa_integrand_old) 
                                      - im_d_kp * (im_y_kappa_integrand + im_y_kappa_integrand_old));
                im_y_result += 0.5 * (re_d_kp * (im_y_kappa_integrand + im_y_kappa_integrand_old) 
                                      + im_d_kp * (re_y_kappa_integrand + re_y_kappa_integrand_old));
                re_z_result += 0.5 * (re_d_kp * (re_z_kappa_integrand + re_z_kappa_integrand_old) 
                                      - im_d_kp * (im_z_kappa_integrand + im_z_kappa_integrand_old));
                im_z_result += 0.5 * (re_d_kp * (im_z_kappa_integrand + im_z_kappa_integrand_old) 
                                      + im_d_kp * (re_z_kappa_integrand + re_z_kappa_integrand_old));
            }
        }
        re_e_x[i] = re_x_result;
        im_e_x[i] = im_x_result;
        re_e_y[i] = re_y_result;
        im_e_y[i] = im_y_result;
        re_e_z[i] = re_z_result;
        im_e_z[i] = im_z_result;
    }
"""
