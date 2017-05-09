import smuthi.index_conversion as idx
import smuthi.nfmds_wrappers

idx.set_swe_specs(l_max=4)

vacuum_wavelength = 550
layer_refractive_index = 1.3
particle_refractive_index = 1.8 + 0.01j
half_axis_z = 100
half_axis_xy = 200
use_ds = True
n_int = 300
n_rank = 8

t = smuthi.nfmds_wrappers.taxsym_tmatrix_spheroid(vacuum_wavelength=vacuum_wavelength,
                                                  layer_refractive_index=layer_refractive_index,
                                                  particle_refractive_index=particle_refractive_index,
                                                  semi_axis_c=half_axis_z, semi_axis_a=half_axis_xy, use_ds=use_ds,
                                                  nint=n_int, nrank=n_rank)


def test_spheroid_tmatrix_against_prototype():
    t00 = -0.416048522578639 + 0.462839918856895j
    assert abs(t[0, 0] - t00) / abs(t00) < 1e-5

    t4210 = -4.663469643281004e-04 - 3.215630661547245e-04j
    assert abs(t[42, 10] - t4210) / abs(t4210) < 1e-5


if __name__ == 'main':
    test_spheroid_tmatrix_against_prototype()
