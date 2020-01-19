import numpy as np
import smuthi.postprocessing.far_field as ff
import smuthi.fields.coordinates_and_contours as coord
from smuthi.fields import angular_frequency


def evaluate(simulation, detector):
    """Run a simulation and evaluate the detector.
    Args:
        simulation (smuthi.simulation.Simulation):    simulation object
        detector (method or str):                     Specify a method that accepts a simulation as input and returns
                                                      a float. Otherwise, type "extinction cross section" to use the
                                                      extinction cross section as a detector.

    Returns:
        The detector value (float)
    """
    if detector == "extinction cross section":
        def detector(sim):
            ecs = ff.extinction_cross_section(initial_field=sim.initial_field,
                                              particle_list=sim.particle_list,
                                              layer_system=sim.layer_system)
            return ecs["top"] + ecs["bottom"]
    simulation.run()
    return detector(simulation)


def converge_l_max(particle,
                   simulation,
                   detector="extinction cross section",
                   tolerance=1e-3,
                   max_iter=100,
                   current_value=None):
    """Find suitable multipole cutoff degree `l_max` for a given particle and simulation. The routine starts with the
    current `l_max` of the particle. The value of `l_max` is successively incremented in a loop until the resulting
    relative change in the detector value is smaller than the specified tolerance. The method updates the input
    particle object with the `l_max` value for which convergence has been achieved.

    Args:
        particle (smuthi.particles.Particle):         Particle for which the l_max is incremented
        simulation (smuthi.simulation.Simulation):    Simulation object containing the particle
        detector (function or string):                Function that accepts a simulation object and returns a detector
                                                      value the change of which is used to define convergence.
                                                      Alternatively, use "extinction cross section" (default) to have
                                                      the extinction cross section as the detector value.
        tolerance (float):                            Relative tolerance for the detector value change.
        max_iter (int):                               Break convergence loop after that number of iterations, even if
                                                      no convergence has been achieved.
        current_value (float):                        Start value of detector (for current settings). If not
                                                      specified the method starts with a simulation for the current
                                                      settings.

    Returns:
        Detector value of converged or break-off parameter settings.
      """
    if current_value is None:
        current_value = evaluate(simulation, detector)

    for _ in range(max_iter):
        old_l_max = particle.l_max
        particle.l_max = old_l_max + 1  # l_max increment
        particle.m_max = particle.l_max

        print("---------------------------------------")
        print("Try l_max = %i and m_max=%i" % (particle.l_max, particle.m_max))

        new_value = evaluate(simulation, detector)
        rel_diff = abs(new_value - current_value) / abs(current_value)
        print("Old detector value:", current_value)
        print("New detector value:", new_value)
        print("Relative difference:", rel_diff)

        if rel_diff < tolerance:  # in this case: discard l_max increment
            particle.l_max = old_l_max
            particle.m_max = old_l_max
            print("Relative difference smaller than tolerance. Keep l_max = %i" % particle.l_max)
            return current_value
        else:
            current_value = new_value

    print("No convergence achieved. Keep l_max = %i"%particle.l_max)
    return current_value

def converge_m_max(particle,
                   simulation,
                   detector="extinction cross section",
                   tolerance=1e-3,
                   current_value=None):
    """Find suitable multipole cutoff order `m_max` for a given particle and simulation. The routine starts with the
    current `l_max` of the particle, i.e. with `m_max=l_max`. The value of `m_max` is successively decremented in a loop
    until the resulting relative change in the detector value is larger than the specified tolerance. The method updates
    the input particle object with the so determined `m_max`.

    Args:
        particle (smuthi.particles.Particle):         Particle for which suitable m_max is searched
        simulation (smuthi.simulation.Simulation):    Simulation object containing the particle
        detector (function or string):                Function that accepts a simulation object and returns a detector
                                                      value the change of which is used to define convergence.
                                                      Alternatively, use "extinction cross section" (default) to have
                                                      the extinction cross section as the detector value.
        tolerance (float):                            Relative tolerance for the detector value change.
        max_iter (int):                               Break convergence loop after that number of iterations, even if
                                                      no convergence has been achieved.
        current_value (float):                        Start value of detector (for current settings). If not
                                                      specified the method starts with a simulation for the current
                                                      settings.

    Returns:
        Detector value of converged or break-off parameter settings.
    """
    if current_value is None:
        current_value = evaluate(simulation, detector)

    for m_max in range(particle.m_max, -1, -1):
        old_m_max = particle.m_max
        particle.m_max = m_max
        print("---------------------------------------")
        print("Try m_max=%i"%particle.m_max)
        new_value = evaluate(simulation, detector)
        rel_diff = abs(new_value - current_value) / abs(current_value)
        print("Old detector value:", current_value)
        print("New detector value:", new_value)
        print("Relative difference:", rel_diff)
        if rel_diff > tolerance:  # in this case: discard m_max decrement
            particle.m_max = old_m_max
            print("Relative difference larger than tolerance. Keep m_max = %i"%particle.m_max)
            return current_value
        else:
            current_value = new_value

    print("No convergence achieved. Keep m_max = %i" % particle.m_max)


def converge_multipole_cutoff(simulation,
                              detector="extinction cross section",
                              tolerance=1e-3,
                              max_iter=100,
                              current_value=None):
    """Find suitable multipole cutoff degree `l_max` and order `m_max` for all particles in a given simulation object.
    The method updates the input simulation object with the so determined multipole truncation values.

    Args:
        simulation (smuthi.simulation.Simulation):    Simulation object
        detector (function or string):                Function that accepts a simulation object and returns a detector
                                                      value the change of which is used to define convergence.
                                                      Alternatively, use "extinction cross section" (default) to have
                                                      the extinction cross section as the detector value.
        tolerance (float):                            Relative tolerance for the detector value change.
        max_iter (int):                               Break convergence loops after that number of iterations, even if
                                                      no convergence has been achieved.
        current_value (float):                        Start value of detector (for current settings). If not
                                                      specified the method starts with a simulation for the current
                                                      settings.

    Returns:
        Detector value of converged or break-off parameter settings.
    """
    for particle in simulation.particle_list:
        current_value = converge_l_max(particle,
                                       simulation,
                                       detector,
                                       tolerance,
                                       max_iter,
                                       current_value)

        current_value = converge_m_max(particle,
                                       simulation,
                                       detector,
                                       tolerance,
                                       current_value)

    return current_value


def neff_waypoints(simulation, neff_imag=1e-2, neff_max=None, neff_max_offset=None):
    """Construct a list of Sommerfeld integral contour waypoints with regard to possible waveguide mode and branchpoint
    singularity locations.

    Args:
        simulation (smuthi.simulation.Simulation):      Simulation object
        neff_imag (float):                              Extent of the contour into the negative imaginary direction
                                                        (in terms of effective refractive index, n_eff=kappa/omega).
        neff_max (float):                               Truncation value of contour (in terms of effective refractive
                                                        index).
        neff_max_offset (float):                        If no value for `neff_max` is specified, use the last estimated
                                                        singularity location plus this value (in terms of effective
                                                        refractive index).

    Returns:
        List of complex waypoint values.
    """

    min_waveguide_neff = max(0, min(np.array(simulation.layer_system.refractive_indices).real) - 0.1)
    max_waveguide_neff = max(np.array(simulation.layer_system.refractive_indices).real) + 0.2
    if neff_max is None:
        if neff_max_offset is None:
            raise ValueError("You need to specify either neff_max or neff_max_offset.")
        else:
            neff_max = max_waveguide_neff + neff_max_offset

    waypoints = [0,
                 min_waveguide_neff,
                 min_waveguide_neff - 1j * neff_imag,
                 max_waveguide_neff - 1j * neff_imag,
                 max_waveguide_neff]
    if neff_max > max_waveguide_neff:
        waypoints.append(neff_max)
    return waypoints


def update_contour(simulation, neff_imag=5e-3, neff_max=None, neff_max_offset=0.5, neff_step=2e-3):
    """Update the `k_parallel` attribute of the input simulation object with a newly constructed Sommerfeld integral
    contour.

    Args:
        simulation (smuthi.simulation.Simulation):      Simulation object
        neff_imag (float):                              Extent of the contour into the negative imaginary direction
                                                        (in terms of effective refractive index, n_eff=kappa/omega).
        neff_max_offset (float):                        Continue the contour by that value after the last estimated
                                                        waveguide mode location (in terms of effective refractive
                                                        index).
        neff_step (float):                              Discretization of the contour (in terms of eff. refractive
                                                        index).
    """
    waypoints = neff_waypoints(simulation, neff_imag, neff_max)
    simulation.k_parallel = coord.complex_contour(simulation.initial_field.vacuum_wavelength, waypoints, neff_step)
    branchpoint_correction(simulation, neff_step * 1e-2)

    waypoints = neff_waypoints(simulation=simulation,
                               neff_imag=neff_imag,
                               neff_max=neff_max,
                               neff_max_offset=neff_max_offset)
    simulation.k_parallel = coord.complex_contour(simulation.initial_field.vacuum_wavelength, waypoints, neff_step)
    branchpoint_correction(simulation, neff_step * 1e-2)


def branchpoint_correction(simulation, neff_minimal_branchpoint_distance):
    """Check if a Sommerfeld integral contour contains possible branchpoint singularities and if so, replace them by
    nearby non-singular locations.

    Args:
        simulation (smuthi.simulation.Simulation):      Simulation object
        neff_minimal_branchpoint_distance (float):      Minimal distance that contour points shall have from
                                                        branchpoint singularities (in terms of effective refractive
                                                        index).
    """

    for n in simulation.layer_system.refractive_indices:
        k = n * angular_frequency(simulation.initial_field.vacuum_wavelength)
        min_k_distance = (angular_frequency(simulation.initial_field.vacuum_wavelength)
                          * neff_minimal_branchpoint_distance)
        while True:
            branchpoint_indices = np.where(abs(simulation.k_parallel - k) < min_k_distance)[0]
            if len(branchpoint_indices) == 0:
                break
            idx = branchpoint_indices[0]
            # replace contour point by two points at the middle towards its left and right neighbors
            if not idx == len(simulation.k_parallel) - 1:
                simulation.k_parallel = np.insert(simulation.k_parallel,
                                                  idx + 1,
                                                  (simulation.k_parallel[idx] + simulation.k_parallel[idx+1]) / 2.0)
                # make sure the new point is ok, otherwise remove
                if abs(simulation.k_parallel[idx + 1] - k) < min_k_distance:
                    simulation.k_parallel = np.delete(simulation.k_parallel, idx + 1)
            if not idx == 0:
                simulation.k_parallel[idx] = (simulation.k_parallel[idx-1] + simulation.k_parallel[idx]) / 2.0
                # make sure the shifted point is ok, otherwise remove
                if abs(simulation.k_parallel[idx] - k) < min_k_distance:
                    simulation.k_parallel = np.delete(simulation.k_parallel, idx)


def converge_neff_max(simulation,
                      detector="extinction cross section",
                      tolerance=1e-3,
                      max_iter=20,
                      neff_imag=1e-2,
                      neff_step=2e-3,
                      neff_max_increment=0.5,
                      converge_lm=True):

    update_contour(simulation=simulation, neff_imag=neff_imag, neff_max_offset=0, neff_step=neff_step)

    neff_max = simulation.k_parallel[-1] / angular_frequency(simulation.initial_field.vacuum_wavelength)

    if converge_lm:
        current_value = converge_multipole_cutoff(simulation=simulation,
                                                  detector=detector,
                                                  tolerance=tolerance,
                                                  max_iter=max_iter)
    else:
        current_value = evaluate(simulation, detector)

    for _ in range(max_iter):
        old_neff_max = neff_max
        neff_max = update_contour(simulation=simulation,
                                  neff_imag=neff_imag,
                                  neff_max=neff_max+neff_max_increment,
                                  neff_step=neff_step)

        print("---------------------------------------")
        print("Try neff_max = %f"%neff_max)

        if converge_lm:
            new_value = converge_multipole_cutoff(simulation=simulation,
                                                  detector=detector,
                                                  tolerance=tolerance,
                                                  max_iter=max_iter,
                                                  current_value=current_value)
        else:
            new_value = evaluate(simulation, detector)

        rel_diff = abs(new_value - current_value) / abs(current_value)
        print("Old detector value:", current_value)
        print("New detector value:", new_value)
        print("Relative difference:", rel_diff)

        if rel_diff < tolerance:  # in this case: discard l_max increment
            neff_max = update_contour(simulation=simulation,
                                      neff_imag=neff_imag,
                                      neff_max=old_neff_max,
                                      neff_step=neff_step)

            print("Relative difference smaller than tolerance. Keep neff_max = %f"%neff_max)
            return current_value
        else:
            current_value = new_value

    print("No convergence achieved. Keep neff_max = %i"%neff_max)
