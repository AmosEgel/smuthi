import numpy as np
import warnings
import sys
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    import pycuda.cumath
    pycuda_available = True
except:
    pycuda_available = False

use_gpu = False


def enable_gpu(enable=True):
    """Sets the use_gpu flag to enable/disable the use of CUDA kernels.

    Args:
        enable (bool): Set use_gpu flag to this value (default=True).
    """
    # todo: do gpuarrays require cuda toolkit? otherwise distinguish if only gpuarrays work but no kernel compilation
    global use_gpu
    
    if use_gpu == enable:
        return
    
    if not enable:
        use_gpu = False
        sys.stdout.write('Disabling GPU usage.\n')
        sys.stdout.flush()
        return
    
    if not pycuda_available:
        warnings.warn("Unable to import PyCuda - fall back to CPU mode")
        use_gpu = False
        sys.stdout.write('Disabling GPU usage.\n')
        sys.stdout.flush()
    else:
        try:
            test_fun = SourceModule("""
                __global__ void test_kernel(float *x)
                {
                    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                    x[i] = 2 * i;
                }""").get_function("test_kernel")
            x = gpuarray.to_gpu(np.zeros(128, dtype=np.float32))
            test_fun(x, block=(128,1,1), grid=(1,1))
            assert np.allclose(x.get(), 2 * np.arange(128))
            use_gpu = True
            sys.stdout.write('Enabling GPU usage.\n')
            sys.stdout.flush()
            return

        except Exception as e:
            warnings.warn("CUDA test kernel failed - fall back to CPU mode")
            print(e)
            use_gpu = False
            sys.stdout.write('Disabling GPU usage.\n')
            sys.stdout.flush()


# The following cuda kernel multiplies the coupling matrix to a vector. It is based on linear interpolation of 
# the lookup table.
#
# input arguments of the cuda kernel:
# n (np.uint32):  n[i] contains the mutlipole multi-index with regard to self.l_max and self.m_max of the i-th
#                 entry of a system vector
# m (np.float32): m[i] contains the multipole order with regard to particle.l_max and particle.m_max
# x_pos (np.float32): x_pos[i] contains the respective particle x-position
# y_pos (np.float32): y_pos[i] contains the respective particle y-position
# z_pos (np.float32): z_pos[i] contains the respective particle z-position
# re_lookup_pl (np.float32): the real part of the lookup table for the z1+z2 part of the Sommerfeld integral,
#                            in the format (rho, sum_z, n1, n2)
# im_lookup_pl (np.float32): the imaginary part of the lookup table for the z1+z2 part of the Sommerfeld
#                            integral, in the format (rho, sum_z, n1, n2)
# re_lookup_mn (np.float32): the real part of the lookup table for the z1-z2 part of the Sommerfeld integral,
#                            in the format (rho, diff_z, n1, n2)
# im_lookup_mn (np.float32): the imaginary part of the lookup table for the z1-z2 part of the Sommerfeld
#                            integral, in the format (rho, diff_z, n1, n2)
# re_in_vec (np.float32): the real part of the vector to be multiplied with the coupling matrix
# im_in_vec (np.float32): the imaginary part of the vector to be multiplied with the coupling matrix
# re_result_vec (np.float32): the real part of the vector into which the result is written
# im_result_vec (np.float32): the imaginary part of the vector into which the result is written
linear_volume_lookup_source = """
    #define BLOCKSIZE %i
    #define NUMBER_OF_UNKNOWNS %i
    #define Z_ARRAY_LENGTH %i
    #define MIN_RHO %f
    #define MIN_Z_SUM %f
    #define MIN_Z_DIFF %f
    #define LOOKUP_RESOLUTION %f
    __global__ void coupling_kernel(const int *n, const float *m, const float *x_pos, const float *y_pos,
                                    const float *z_pos, const float *re_lookup_pl, const float *im_lookup_pl,
                                    const float *re_lookup_mn, const float *im_lookup_mn,
                                    const float *re_in_vec, const float *im_in_vec,
                                    float  *re_result, float  *im_result)
    {
        unsigned int i1 = blockIdx.x * blockDim.x + threadIdx.x;
        if(i1 >= NUMBER_OF_UNKNOWNS) return;

        const float x1 = x_pos[i1];
        const float y1 = y_pos[i1];
        const float z1 = z_pos[i1];

        const int n1 = n[i1];
        const float m1 = m[i1];

        re_result[i1] = 0.0;
        im_result[i1] = 0.0;
        
        for (int i2=0; i2<NUMBER_OF_UNKNOWNS; i2++)
        {
            float x21 = x1 - x_pos[i2];
            float y21 = y1 - y_pos[i2];
            float sz21 = z1 + z_pos[i2];
            float dz21 = z1 - z_pos[i2];

            const int n2 = n[i2];
            const float m2 = m[i2];

            float rho = sqrt(x21*x21+y21*y21);
            float phi = atan2(y21,x21);
            
            int rho_idx = (int) floor((rho - MIN_RHO) / LOOKUP_RESOLUTION);
            float rho_w = (rho - MIN_RHO) / LOOKUP_RESOLUTION - floor((rho - MIN_RHO) / LOOKUP_RESOLUTION);
            
            int sz_idx = (int) floor((sz21 - MIN_Z_SUM) / LOOKUP_RESOLUTION);
            float sz_w = (sz21 - MIN_Z_SUM) / LOOKUP_RESOLUTION - floor((sz21 - MIN_Z_SUM) / LOOKUP_RESOLUTION);

            int dz_idx = (int) floor((dz21 - MIN_Z_DIFF) / LOOKUP_RESOLUTION);
            float dz_w = (dz21 - MIN_Z_DIFF) / LOOKUP_RESOLUTION 
                            - floor((dz21 - MIN_Z_DIFF) /  LOOKUP_RESOLUTION);
            
            int idx_rho_sz = rho_idx * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE 
                                +  sz_idx * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;

            int idx_rho_szpl1 = rho_idx * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE
                                + (sz_idx + 1) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;

            int idx_rhopl1_sz = (rho_idx + 1) * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE
                                + sz_idx * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;

            int idx_rhopl1_szpl1 = (rho_idx + 1) * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE
                                    + (sz_idx + 1) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;

            int idx_rho_dz = rho_idx * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE + dz_idx * BLOCKSIZE * BLOCKSIZE
                                + n1 * BLOCKSIZE + n2;

            int idx_rho_dzpl1 = rho_idx * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE
                                + (dz_idx + 1) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;

            int idx_rhopl1_dz = (rho_idx + 1) * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE
                                + dz_idx * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;

            int idx_rhopl1_dzpl1 = (rho_idx + 1) * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE
                                    + (dz_idx + 1) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;
            
            float f_rhoi = re_lookup_pl[idx_rho_sz] * (1 - sz_w) + re_lookup_pl[idx_rho_szpl1] * sz_w;
            float f_rhoipl1 = re_lookup_pl[idx_rhopl1_sz] * (1 - sz_w) + re_lookup_pl[idx_rhopl1_szpl1] * sz_w;
            float re_si_pl = f_rhoi * (1 - rho_w) + f_rhoipl1 * rho_w;

            f_rhoi = im_lookup_pl[idx_rho_sz] * (1 - sz_w) + im_lookup_pl[idx_rho_szpl1] * sz_w;
            f_rhoipl1 = im_lookup_pl[idx_rhopl1_sz] * (1 - sz_w) + im_lookup_pl[idx_rhopl1_szpl1] * sz_w;
            float im_si_pl = f_rhoi * (1 - rho_w) + f_rhoipl1 * rho_w;
            
            f_rhoi = re_lookup_mn[idx_rho_dz] * (1 - dz_w) + re_lookup_mn[idx_rho_dzpl1] * dz_w;
            f_rhoipl1 = re_lookup_mn[idx_rhopl1_dz] * (1 - dz_w) + re_lookup_mn[idx_rhopl1_dzpl1] * dz_w;
            float re_si_mn = f_rhoi * (1 - rho_w) + f_rhoipl1 * rho_w;

            f_rhoi = im_lookup_mn[idx_rho_dz] * (1 - dz_w) + im_lookup_mn[idx_rho_dzpl1] * dz_w;
            f_rhoipl1 = im_lookup_mn[idx_rhopl1_dz] * (1 - dz_w) + im_lookup_mn[idx_rhopl1_dzpl1] * dz_w;
            float im_si_mn = f_rhoi * (1 - rho_w) + f_rhoipl1 * rho_w;
            
            float re_eimphi = cosf((m2 - m1) * phi);
            float im_eimphi = sinf((m2 - m1) * phi);
            
            float re_w = re_eimphi * (re_si_pl + re_si_mn) - im_eimphi * (im_si_pl + im_si_mn);
            float im_w = im_eimphi * (re_si_pl + re_si_mn) + re_eimphi * (im_si_pl + im_si_mn);
            
            re_result[i1] += re_w * re_in_vec[i2] - im_w * im_in_vec[i2];
            im_result[i1] += re_w * im_in_vec[i2] + im_w * re_in_vec[i2];
        }
    }"""
            
# The following cuda kernel multiplies the coupling matrix to a vector. It is based on cubic Hermite spline interpolation
# of the lookup table.
#
# input arguments of the cuda kernel:
# n (np.uint32):  n[i] contains the mutlipole multi-index with regard to self.l_max and self.m_max of the i-th
#                 entry of a system vector
# m (np.float32): m[i] contains the multipole order with regard to particle.l_max and particle.m_max
# x_pos (np.float32): x_pos[i] contains the respective particle x-position
# y_pos (np.float32): y_pos[i] contains the respective particle y-position
# z_pos (np.float32): z_pos[i] contains the respective particle z-position
# re_lookup_pl (np.float32): the real part of the lookup table for the z1+z2 part of the Sommerfeld integral,
#                            in the format (rho, sum_z, n1, n2)
# im_lookup_pl (np.float32): the imaginary part of the lookup table for the z1+z2 part of the Sommerfeld
#                            integral, in the format (rho, sum_z, n1, n2)
# re_lookup_mn (np.float32): the real part of the lookup table for the z1-z2 part of the Sommerfeld integral,
#                            in the format (rho, diff_z, n1, n2)
# im_lookup_mn (np.float32): the imaginary part of the lookup table for the z1-z2 part of the Sommerfeld
#                            integral, in the format (rho, diff_z, n1, n2)
# re_in_vec (np.float32): the real part of the vector to be multiplied with the coupling matrix
# im_in_vec (np.float32): the imaginary part of the vector to be multiplied with the coupling matrix
# re_result_vec (np.float32): the real part of the vector into which the result is written
# im_result_vec (np.float32): the imaginary part of the vector into which the result is written
cubic_volume_lookup_source = """
    #define BLOCKSIZE %i
    #define NUMBER_OF_UNKNOWNS %i
    #define Z_ARRAY_LENGTH %i
    #define MIN_RHO %f
    #define MIN_Z_SUM %f
    #define MIN_Z_DIFF %f
    #define LOOKUP_RESOLUTION %f
    
    __device__ int lookup_index(int const i_r, int const i_z, int const n1, int const n2)
    {
        return i_r * Z_ARRAY_LENGTH * BLOCKSIZE * BLOCKSIZE + i_z * BLOCKSIZE * BLOCKSIZE + n1 *  BLOCKSIZE + n2;
    }
    
    __device__ float cubic_interpolation(float const w, float const lookup_imn1, float const lookup_i, 
                                            float const lookup_ipl1, float const lookup_ipl2)
    {
        return ((-w*w*w+2*w*w-w) * lookup_imn1 + (3*w*w*w-5*w*w+2) * lookup_i + (-3*w*w*w+4*w*w+w) * lookup_ipl1
                + (w*w*w-w*w) * lookup_ipl2) / 2;
    }
    
    __device__ float lookup_2D(int const i_rho, float const w_rho, int const i_z, float const w_z, int const n1,
                                int const n2, float const *lookup)
    {
        int i_zmn1 = lookup_index(i_rho-1, i_z-1, n1, n2);
        int i = lookup_index(i_rho-1, i_z, n1, n2);
        int i_zpl1 = lookup_index(i_rho-1, i_z+1, n1, n2);
        int i_zpl2 = lookup_index(i_rho-1, i_z+2, n1, n2);
        float lookup_rmn1 = cubic_interpolation(w_z, lookup[i_zmn1], lookup[i], lookup[i_zpl1], lookup[i_zpl2]);
        
        i_zmn1 = lookup_index(i_rho, i_z-1, n1, n2);
        i = lookup_index(i_rho, i_z, n1, n2);
        i_zpl1 = lookup_index(i_rho, i_z+1, n1, n2);
        i_zpl2 = lookup_index(i_rho, i_z+2, n1, n2);
        float lookup_r = cubic_interpolation(w_z, lookup[i_zmn1], lookup[i], lookup[i_zpl1], lookup[i_zpl2]);
        
        i_zmn1 = lookup_index(i_rho+1, i_z-1, n1, n2);
        i = lookup_index(i_rho+1, i_z, n1, n2);
        i_zpl1 = lookup_index(i_rho+1, i_z+1, n1, n2);
        i_zpl2 = lookup_index(i_rho+1, i_z+2, n1, n2);
        float lookup_rpl1 = cubic_interpolation(w_z, lookup[i_zmn1], lookup[i], lookup[i_zpl1], lookup[i_zpl2]);
        
        i_zmn1 = lookup_index(i_rho+2, i_z-1, n1, n2);
        i = lookup_index(i_rho+2, i_z, n1, n2);
        i_zpl1 = lookup_index(i_rho+2, i_z+1, n1, n2);
        i_zpl2 = lookup_index(i_rho+2, i_z+2, n1, n2);
        float lookup_rpl2 = cubic_interpolation(w_z, lookup[i_zmn1], lookup[i], lookup[i_zpl1], lookup[i_zpl2]);
        
        return cubic_interpolation(w_rho, lookup_rmn1, lookup_r, lookup_rpl1, lookup_rpl2);
    }
    
    __global__ void coupling_kernel(const int *n, const float *m, const float *x_pos, const float *y_pos,
                                    const float *z_pos, const float *re_lookup_pl, const float *im_lookup_pl,
                                    const float *re_lookup_mn, const float *im_lookup_mn,
                                    const float *re_in_vec, const float *im_in_vec,
                                    float  *re_result, float  *im_result)
    {
        unsigned int i1 = blockIdx.x * blockDim.x + threadIdx.x;
        if(i1 >= NUMBER_OF_UNKNOWNS) return;

        const float x1 = x_pos[i1];
        const float y1 = y_pos[i1];
        const float z1 = z_pos[i1];

        const int n1 = n[i1];
        const float m1 = m[i1];

        re_result[i1] = 0.0;
        im_result[i1] = 0.0;
        
        for (int i2=0; i2<NUMBER_OF_UNKNOWNS; i2++)
        {
            float x21 = x1 - x_pos[i2];
            float y21 = y1 - y_pos[i2];
            float sz21 = z1 + z_pos[i2];
            float dz21 = z1 - z_pos[i2];

            const int n2 = n[i2];
            const float m2 = m[i2];

            float rho = sqrt(x21*x21+y21*y21);
            float phi = atan2(y21,x21);
            
            int rho_idx = (int) floor((rho - MIN_RHO) / LOOKUP_RESOLUTION);
            float rho_w = (rho - MIN_RHO) / LOOKUP_RESOLUTION - floor((rho - MIN_RHO) / LOOKUP_RESOLUTION);
            
            int sz_idx = (int) floor((sz21 - MIN_Z_SUM) / LOOKUP_RESOLUTION);
            float sz_w = (sz21 - MIN_Z_SUM) / LOOKUP_RESOLUTION - floor((sz21 - MIN_Z_SUM) / LOOKUP_RESOLUTION);

            int dz_idx = (int) floor((dz21 - MIN_Z_DIFF) / LOOKUP_RESOLUTION);
            float dz_w = (dz21 - MIN_Z_DIFF) / LOOKUP_RESOLUTION 
                            - floor((dz21 - MIN_Z_DIFF) /  LOOKUP_RESOLUTION);
            
            float re_si_pl = lookup_2D(rho_idx, rho_w, sz_idx, sz_w, n1, n2, re_lookup_pl);
            float im_si_pl = lookup_2D(rho_idx, rho_w, sz_idx, sz_w, n1, n2, im_lookup_pl);
            float re_si_mn = lookup_2D(rho_idx, rho_w, dz_idx, dz_w, n1, n2, re_lookup_mn);
            float im_si_mn = lookup_2D(rho_idx, rho_w, dz_idx, dz_w, n1, n2, im_lookup_mn);
            
            float re_eimphi = cosf((m2 - m1) * phi);
            float im_eimphi = sinf((m2 - m1) * phi);
            
            float re_w = re_eimphi * (re_si_pl + re_si_mn) - im_eimphi * (im_si_pl + im_si_mn);
            float im_w = im_eimphi * (re_si_pl + re_si_mn) + re_eimphi * (im_si_pl + im_si_mn);
            
            re_result[i1] += re_w * re_in_vec[i2] - im_w * im_in_vec[i2];
            im_result[i1] += re_w * im_in_vec[i2] + im_w * re_in_vec[i2];
        }
    }"""


# This cuda kernel multiplies the coupling matrix to a vector. It is based on linear interpolation of the lookup
# table.
#
# input arguments of the cuda kernel:
# n (np.uint32):  n[i] contains the mutlipole multi-index with regard to self.l_max and self.m_max of the i-th
#                 entry of a system vector
# m (np.float32): m[i] contains the multipole order with regard to particle.l_max and particle.m_max
# x_pos (np.float32): x_pos[i] contains the respective particle x-position
# y_pos (np.float32): y_pos[i] contains the respective particle y-position
# re_lookup (np.float32): the real part of the lookup table, in the format [r, n1, n2]
# im_lookup (np.float32): the imaginary part of the lookup table, in the format [r, n1, n2]
# re_in_vec (np.float32): the real part of the vector to be multiplied with the coupling matrix
# im_in_vec (np.float32): the imaginary part of the vector to be multiplied with the coupling matrix
# re_result_vec (np.float32): the real part of the vector into which the result is written
# im_result_vec (np.float32): the imaginary part of the vector into which the result is written
linear_radial_lookup_source = """
    #define BLOCKSIZE %i
    #define NUMBER_OF_UNKNOWNS %i
    #define MIN_RHO %f
    #define LOOKUP_RESOLUTION %f
    
    __global__ void coupling_kernel(const int *n, const float *m, const float *x_pos, const float *y_pos,
                                    const float *re_lookup, const float *im_lookup, const float *re_in_vec,
                                    const float *im_in_vec, float  *re_result, float  *im_result)
    {
        unsigned int i1 = blockIdx.x * blockDim.x + threadIdx.x;
        if(i1 >= NUMBER_OF_UNKNOWNS) return;
        
        const float x1 = x_pos[i1];
        const float y1 = y_pos[i1];
        
        const int n1 = n[i1];
        const float m1 = m[i1];

        re_result[i1] = 0.0;
        im_result[i1] = 0.0;
        
        for (int i2=0; i2<NUMBER_OF_UNKNOWNS; i2++)
        {
            float x21 = x1 - x_pos[i2];
            float y21 = y1 - y_pos[i2];
            
            const int n2 = n[i2];
            const float m2 = m[i2];
            
            float r = sqrt(x21*x21+y21*y21);
            float phi = atan2(y21,x21);
        
            int r_idx = (int) floor((r - MIN_RHO) / LOOKUP_RESOLUTION);
            float w = (r - MIN_RHO) / LOOKUP_RESOLUTION - floor((r - MIN_RHO) / LOOKUP_RESOLUTION);
            
            int idx = r_idx * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;
            int idx_pl_1 = (r_idx + 1) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;
            
            float re_si = re_lookup[idx] + w * (re_lookup[idx_pl_1] - re_lookup[idx]);
            float im_si = im_lookup[idx] + w * (im_lookup[idx_pl_1] - im_lookup[idx]);
            
            float re_eimphi = cosf((m2 - m1) * phi);
            float im_eimphi = sinf((m2 - m1) * phi);
        
            float re_w = re_eimphi * re_si - im_eimphi * im_si;
            float im_w = im_eimphi * re_si + re_eimphi * im_si;
            
            re_result[i1] += re_w * re_in_vec[i2] - im_w * im_in_vec[i2];
            im_result[i1] += re_w * im_in_vec[i2] + im_w * re_in_vec[i2];
            
        }
    }"""


# This cuda kernel multiplies the coupling matrix to a vector. It is based on cubic Hermite spline interpolation of the 
# lookup table.
#
# input arguments of the cuda kernel:
# n (np.uint32):  n[i] contains the mutlipole multi-index with regard to self.l_max and self.m_max of the i-th
#                 entry of a system vector
# m (np.float32): m[i] contains the multipole order with regard to particle.l_max and particle.m_max
# x_pos (np.float32): x_pos[i] contains the respective particle x-position
# y_pos (np.float32): y_pos[i] contains the respective particle y-position
# re_lookup (np.float32): the real part of the lookup table, in the format [r, n1, n2]
# im_lookup (np.float32): the imaginary part of the lookup table, in the format [r, n1, n2]
# re_in_vec (np.float32): the real part of the vector to be multiplied with the coupling matrix
# im_in_vec (np.float32): the imaginary part of the vector to be multiplied with the coupling matrix
# re_result_vec (np.float32): the real part of the vector into which the result is written
# im_result_vec (np.float32): the imaginary part of the vector into which the result is written
cubic_radial_lookup_source = """
    #define BLOCKSIZE %i
    #define NUMBER_OF_UNKNOWNS %i
    #define MIN_RHO %f
    #define LOOKUP_RESOLUTION %f
    
    __device__ float cubic_interpolation(float const w, float const lookup_imn1, float const lookup_i,
                                         float const lookup_ipl1, float const lookup_ipl2)
    {
        return ((-w*w*w+2*w*w-w) * lookup_imn1 + (3*w*w*w-5*w*w+2) * lookup_i + (-3*w*w*w+4*w*w+w) * lookup_ipl1
                + (w*w*w-w*w) * lookup_ipl2) / 2;
    }

    
    __global__ void coupling_kernel(const int *n, const float *m, const float *x_pos, const float *y_pos,
                                    const float *re_lookup, const float *im_lookup, const float *re_in_vec,
                                    const float *im_in_vec, float  *re_result, float  *im_result)
    {
        unsigned int i1 = blockIdx.x * blockDim.x + threadIdx.x;
        if(i1 >= NUMBER_OF_UNKNOWNS) return;
        
        const float x1 = x_pos[i1];
        const float y1 = y_pos[i1];
        
        const int n1 = n[i1];
        const float m1 = m[i1];

        re_result[i1] = 0.0;
        im_result[i1] = 0.0;
        
        for (int i2=0; i2<NUMBER_OF_UNKNOWNS; i2++)
        {
            float x21 = x1 - x_pos[i2];
            float y21 = y1 - y_pos[i2];
            
            const int n2 = n[i2];
            const float m2 = m[i2];
            
            float r = sqrt(x21*x21+y21*y21);
            float phi = atan2(y21,x21);
        
            int r_idx = (int) floor((r - MIN_RHO) / LOOKUP_RESOLUTION);
            float w = (r - MIN_RHO) / LOOKUP_RESOLUTION - floor((r - MIN_RHO) / LOOKUP_RESOLUTION);
            
            int idx_mn1 = (r_idx - 1) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;
            int idx = r_idx * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;
            int idx_pl1 = (r_idx + 1) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;
            int idx_pl2 = (r_idx + 2) * BLOCKSIZE * BLOCKSIZE + n1 * BLOCKSIZE + n2;
            
            float re_si = cubic_interpolation(w, re_lookup[idx_mn1], re_lookup[idx], re_lookup[idx_pl1], 
                                              re_lookup[idx_pl2]);
            float im_si = cubic_interpolation(w, im_lookup[idx_mn1], im_lookup[idx], im_lookup[idx_pl1], 
                                              im_lookup[idx_pl2]);                                                

            float re_eimphi = cosf((m2 - m1) * phi);
            float im_eimphi = sinf((m2 - m1) * phi);
        
            float re_w = re_eimphi * re_si - im_eimphi * im_si;
            float im_w = im_eimphi * re_si + re_eimphi * im_si;
            
            re_result[i1] += re_w * re_in_vec[i2] - im_w * im_in_vec[i2];
            im_result[i1] += re_w * im_in_vec[i2] + im_w * re_in_vec[i2];
        }
    }"""


# This cuda kernel is used for the calculation of volume lookup tables (see smuthi.particle_coupling module).
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

# This cuda kernel is used for the calculation of radial lookup tables (see smuthi.particle_coupling module).
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
