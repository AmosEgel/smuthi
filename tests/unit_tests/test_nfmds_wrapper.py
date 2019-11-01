import smuthi.linearsystem.tmatrix.nfmds.t_matrix_axsym as taxs

vacuum_wavelength = 550
layer_refractive_index = 1.3
particle_refractive_index = 1.8 + 0.01j
half_axis_z = 100
half_axis_xy = 200
use_ds = True
n_int = 300
n_rank = 8

cylinder_height = 100
cylinder_radius = 200

t_s = taxs.tmatrix_spheroid(vacuum_wavelength=vacuum_wavelength,
                            layer_refractive_index=layer_refractive_index,
                            particle_refractive_index=particle_refractive_index,
                            semi_axis_c=half_axis_z, semi_axis_a=half_axis_xy, use_ds=use_ds,
                            nint=n_int, nrank=n_rank, l_max=4, m_max=4)

t_c = taxs.tmatrix_cylinder(vacuum_wavelength=vacuum_wavelength,
                            layer_refractive_index=layer_refractive_index,
                            particle_refractive_index=particle_refractive_index,
                            cylinder_height=cylinder_height, cylinder_radius=cylinder_radius,
                            use_ds=use_ds, nint=n_int, nrank=n_rank, l_max=4, m_max=4)


def test_spheroid_tmatrix_against_prototype():
    t00 = -0.416048522578639 + 0.462839918856895j
    assert abs(t_s[0, 0] - t00) / abs(t00) < 1e-5

    t4210 = -4.663469643281004e-04 - 3.215630661547245e-04j
    assert abs(t_s[42, 10] - t4210) / abs(t4210) < 1e-5


def test_cylinder_tmatrix_against_prototype():
    t00 = -0.119828956584570 + 0.282351044628953j
    assert abs(t_c[0, 0] - t00) / abs(t00) < 1e-5

    t4210 = -0.001017849863151 - 0.000754036833086j
    assert abs(t_c[42, 10] - t4210) / abs(t4210) < 1e-5


if __name__ == '__main__':
    test_spheroid_tmatrix_against_prototype()
    test_cylinder_tmatrix_against_prototype()
