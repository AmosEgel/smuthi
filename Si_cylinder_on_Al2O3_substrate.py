# This is an exemplary script to run SMUTHI from within python.
#
# It evaluates the scattering response of a gold NP
# in vacuum. The system is excited by a plane wave.
from mpi4py import MPI
import numpy as np
import sys
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
from_WL = 400 #nm
#from_WL = 570  #nm
to_WL = 1100  #nm<

from_cylinder_R = 100 #nm
#from_cylinder_R = 250 #nm
to_cylinder_R = 515 #nm
cylinder_height = 640 #nm

# total_WL_points = 150*2
# total_R_points = 70*3
total_WL_points = 32
total_R_points = 12
#suffix = "-aSi-n301"
l_max = 10
neff_max = 1.5
# l_max = 15
# neff_max = 10

polar_ang= np.pi/2-np.pi/7.2,  # 25 grad to the surface
#polar_ang= 0

#suffix = "-a-Si-n"+str(total_WL_points)+"-l"+str(l_max)+"-neff"+str(neff_max)
suffix = "-test-Si-fixed-cylH"+str(cylinder_height)+"-n"+str(total_WL_points)+"-l"+str(l_max)+"-neff"+str(neff_max)+"-polar%g"%(polar_ang)
#suffix = "-Si-Green-cylH"+str(cylinder_height)+"-n"+str(total_WL_points)+"-l"+str(l_max)+"-neff"+str(neff_max)
file_ext = ".pdf"

n_air = 1.0
 
collect_NA = 0.42 # Collect objective apperture
integral_samples = 180 

TE = 0
TM = 1
TEM = 2
output_directory = 'smuthi_output'



import matplotlib.pyplot as plt
import matplotlib.colors as mc
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.coordinates
import smuthi.cuda_sources
import smuthi.scattered_field
import smuthi.graphical_output
from smuthi.optical_constants import read_refractive_index_from_yaml as GetIndex
import sys
import subprocess

#smuthi.cuda_sources.enable_gpu()  # Enable GPU acceleration (if available)
def read_zarina():
    zdata = np.genfromtxt(output_directory+"/m0_forpython.txt", delimiter=" "
                          ,usecols=(0,3))
    print(zdata[:,0])
def plot_file():
    filename = output_directory+"/data_TM"+suffix+".txt"
    data = np.loadtxt(filename)
    dataTM = data
    combined_plot(data,TM)
    # # try:
    # filename = output_directory+"/data_TE"+suffix+".txt"
    # data = np.loadtxt(filename)
    # dataTE = data
    # combined_plot(data,TE)
    # # except:
    # #     pass
    # # try:

    # dataTE[:,2:3] += dataTM[:,2:3]
    # combined_plot(dataTE,TEM)

    # except:
    #     pass
    
def combined_plot(data, polarization):
    min_Q = 20
    max_Q = 1e3
    mval = "m1"
    zdata = np.genfromtxt(output_directory+"/"+mval+"_forpython.txt", delimiter=" "
                          ,usecols=(0,3,4))
    cmax = zdata[:,2] < max_Q
    cmin = zdata[:,2] > min_Q
    condition = [[a and b]*3 for a,b in zip(cmin,cmax)]
    zdata = np.reshape(np.extract(condition, zdata),(-1,3))

    sign = "TE"
    if polarization == TM: sign = "TM"
    if polarization == TEM: sign = "TEM"
    np.savetxt(output_directory+"/data_"+sign+".txt",data,header="WL cylinder_R Q_sca Q_ext")
    np.savetxt(output_directory+"/data_"+sign+suffix+".txt",data,header="WL cylinder_R Q_sca Q_ext")
    print(data)
    WLs = np.unique(data[:,0])
    Ds = np.unique(data[:,1])*2
    # # Rescale to better show the axes
    # scale_x = np.linspace(
    #     min(coordX) , max(coordX), len(WLs))
    # scale_z = np.linspace(
    #     min(coordZ) * WL / 2.0 / np.pi, max(coordZ) * WL / 2.0 / np.pi, npts)
    none_line = np.array([np.NaN] * len(WLs))
    Qsca = data[:,2].reshape(len(WLs),len(Ds)).T
    #Qsca = np.vstack((Qsca,none_line,none_line))/1e6
    Qsca = Qsca/1e6
    #print(Qsca)
    Qext = data[:,3].reshape(len(WLs),len(Ds)).T
    Qlist = [Qsca,Qext]
    Qname = ["Qsca",'Qext']
    #vscale = [0.0004,0.2]
    vscale = [0.001,0.2]
    min_tick = [1.0*10**5, 10**5]
    for q, name, vscale,min_tick in list(zip(Qlist,Qname,vscale,min_tick)):
        # norm = mc.Normalize()
        # if name == 'Qsca':
        #fig, ax = plt.subplots(figsize=(6,6))
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(zdata[:,1],zdata[:,0], c="white", alpha=0.8, s=(zdata[:,2])/50, marker="o", linewidth = 0)
        #max_tick = np.amax(q[~np.isnan(q)])*vscale
        min_tick = 0.1
        #max_tick = 4.3
        #max_tick = 3.7
        #max_tick = 2.2
        max_tick = 1.8
        scale_ticks = np.linspace(min_tick, max_tick, 5)

        norm = mc.LogNorm()
        cax = ax.imshow(q
                        , interpolation='nearest'
                        #, interpolation='quadric'
                        # , cmap='jet',
                        #, cmap='gist_ncar',
                        #, cmap='rainbow',
                        , cmap='jet',
                        origin='lower'
                        , vmin=min_tick
                        , vmax=max_tick
                            , extent=(min(WLs), max(WLs), min(Ds), max(Ds))
                        #    ,norm = norm
                        )
        ax.set_aspect(1)
        # Add colorbar
        cbar = fig.colorbar(cax, ticks=[a for a in scale_ticks], ax=ax, fraction=0.046, pad=0.04)

        plt.title(name+"_"+sign+suffix+"\n"+mval+" filtered:  "+str(min_Q)+" < Q < "+str(max_Q))
        plt.xlabel(r'$Z,\lambda$ nm')
        plt.ylabel(r'$D,nm$')

        # plt.axis("image")
        #plt.savefig(output_directory+"/plot_"+name+"_"+sign+suffix+'_log1'+file_ext)
        plt.savefig(output_directory+"/plot_"+name+"_"+sign+suffix+"_"+mval+'_norm'+file_ext)
        plt.close()
    # 2D plot
    if (False):
        max_tick = [25,8]
        for q, name, max_tick in list(zip(Qlist,Qname,max_tick)):
            fig, ax = plt.subplots(figsize=(6,6))
            scale = 10**6
            for i in range(len(Ds)):
                ax.plot(WLs,q[i]/scale,label=str(Ds[i]))
            # cax = ax.imshow(q
            #                     , extent=(min(WLs), max(WLs), min(Ds), max(Ds))
            legend = ax.legend()
            ax.set_ylim(0,max_tick)
            plt.title(r"$\alpha-Si$ (Pierce-Palik) cylinder on $SiO_2$ (Gao) substrate")
            plt.xlabel(r'$\lambda$, nm')
            plt.ylabel(name+r'$, \mu m^2$')
            plt.savefig(output_directory+"/plot1D_"+name+"_"+sign+suffix+file_ext)
            # plt.savefig(output_directory+"/plot1D_"+name+"_"+sign+suffix+file_ext)
            plt.close()


def PrintProgress(progress):
    if mpi_rank == 0:
        sys.__stdout__.write( str(int(progress*10)/10.0)+"%\n")
        sys.__stdout__.flush()
        if sys.platform.startswith('win'):
            subprocess.run('C:\\WINDOWS\\system32\\WindowsPowerShell\\v1.0\\powershell.exe Add-PSSnapin Microsoft.HPC; set-HpcJob -id %CCP_JOBID% -progress '+str(int(progress)), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def GetTopTSCS(WL, index_NP, index_substrate, index_substrate2, cylinder_R, polarization):
    spacer = 2 #nm
    # Initialize a plane wave object the initial field
    global polar_ang
    gaussian_beam = smuthi.initial_field.PlaneWave(
        vacuum_wavelength=WL,
        polar_angle= np.pi/2-np.pi/7.2,  # 25 grad to the surface
        #polar_angle= polar_ang,  
        azimuthal_angle=0,
        polarization=polarization)           # 0 stands for TE, 1 stands for TM
    pol_name = "TE"
    if polarization == 1: pol_name = "TM"

    # Initialize the layer system object
    #The coordinate system is such that the interface between the
    # first two layers defines the plane z=0.

    # two_layers = smuthi.layers.LayerSystem(
    #     thicknesses=[0,  3000, 0],         #  ambient, substrate
    #     refractive_indices=[n_air, index_substrate, index_substrate2])   # like air,glass

    two_layers = smuthi.layers.LayerSystem(
        thicknesses=[0, 0],         #  ambient, substrate
        refractive_indices=[n_air, index_substrate])   # like air,glass

    # Define the scattering particles
    particle_grid = []

    t_matrix_method = {'use discrete sources': False}#, 'nint': 1000, 'nrank': 50}
    
    cylinder = smuthi.particles.FiniteCylinder(
        position=[0, 0, -cylinder_height/2.0-spacer],
        refractive_index=index_NP,
        t_matrix_method = t_matrix_method,
        cylinder_radius = cylinder_R,
        cylinder_height = cylinder_height,
        l_max=l_max)    # choose l_max with regard to particle size and
                    # material higher means more accurate but slower)
    
    # sphere = smuthi.particles.Sphere(position=[0, 0, -core_r-spacer],
    #                                  refractive_index=index_NP,
    #                                  radius=core_r,
    #                                  l_max=3)    # choose l_max with regard to particle size and material
                                                 # higher means more accurate but slower
    particle_grid.append(cylinder)

    # Define contour for Sommerfeld integral
    smuthi.coordinates.set_default_k_parallel(vacuum_wavelength=gaussian_beam.vacuum_wavelength,
                                              neff_resolution=5e-3,       # smaller value means more accurate but slower
                                              neff_max=neff_max)                 # should be larger than the highest refractive
                                                                          # index of the layer system

    # Initialize and run simulation
    simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                              particle_list=particle_grid,
                                              initial_field=gaussian_beam,
                                              solver_type='LU',
                                              # solver_type='gmres',
                                              solver_tolerance=1e-5,
                                              store_coupling_matrix=True,
                                              coupling_matrix_lookup_resolution=None,
                                              # store_coupling_matrix=False,
                                              # coupling_matrix_lookup_resolution=5,
                                              coupling_matrix_interpolator_kind='cubic',
                                              log_to_file=False
                                              ,log_to_terminal = False
                                                  )
    simulation.run()
    


#    p_angles = np.linspace(0, np.pi, integral_samples, dtype=float)
    collect_angle = np.arcsin(collect_NA/n_air)
    #print("Collecting angle for output objective =", collect_angle)
    
    p_angles = np.linspace(np.pi-collect_angle, np.pi, integral_samples, dtype=float)
    a_angles = np.linspace(0, 2.0*np.pi, integral_samples, dtype=float)
    #print(angles)
    scattering_cross_section = smuthi.scattered_field.scattering_cross_section(
        initial_field=gaussian_beam,
        particle_list=particle_grid,
        layer_system=two_layers
        # ,polar_angles=p_angles
        # ,azimuthal_angles=a_angles
        )
    extinction_cross_section = smuthi.scattered_field.extinction_cross_section(
        initial_field=gaussian_beam, particle_list=particle_grid,
        layer_system=two_layers)

    Q_sca = (#scattering_cross_section.top().integral()# [0]
            # + scattering_cross_section.top().integral()[1]
                 scattering_cross_section.bottom().integral()[0]
                 + scattering_cross_section.bottom().integral()[1]
    ).real
    Q_ext = (extinction_cross_section['top'] + extinction_cross_section['bottom']).real

    plot_size = WL/2/2

    if total_WL_points == 1:
        smuthi.graphical_output.show_near_field(
            # quantities_to_plot=[ 'norm_E', 'norm_E_scat', 'E_x','H_x','E_y','H_y','E_z','H_z'],
            quantities_to_plot=[ 'E_x'],
                                        save_plots=True,
                                        show_plots=False,
                                        save_animations=True,
                                        outputdir=output_directory+'/near_field_plots_xz_'
                +pol_name,
                                        xmin=-5*plot_size,
                                        xmax=5*plot_size,
                                        ymin=0, #-2*plot_size,
                                        ymax=0, #-2*plot_size,
                                        zmin=-5*plot_size,
                                        zmax=+plot_size*2,
                                        # resolution_step=plot_size/4.0,
                                        resolution_step=WL/5.0/2,
                                        interpolate_step=None,
                                        interpolation_order=1,
                                        simulation=simulation
                                         , max_field=0.5
                                        # , min_norm_field=0.87
                                            )

    return WL, cylinder_R, Q_sca, Q_ext

def call_worker(args):
    #print(args)
    result =  GetTopTSCS(args[0],args[1],args[2],args[3],args[4],args[5])
    return result

def main(polarization):
    WLs = np.linspace(from_WL, to_WL, total_WL_points)
    cylinder_Rs = np.linspace(from_cylinder_R,
                              to_cylinder_R, total_R_points)

    #index_Si = GetIndex('data/a-Si-Pierce-Palik.yml.2', WLs, "nm")
    index_Si = GetIndex('data/Si-Green-2008.yml', WLs, "nm")
    #index_Si = GetIndex('data/Si-Aspnes.yml', WLs, "nm")
    #index_Au = GetIndex('data/Au-Rakic-LD.yml', WLs, "nm")
    #index_Au = GetIndex('data/Au-Johnson.yml', WLs, "nm")
    #index_substrate = GetIndex('data/Al2O3-Sapphire-ordinary-Querry.yml', WLs, "nm")
    #index_substrate = GetIndex('data/SiO2-Gao.yml', WLs, "nm")
    index_substrate = GetIndex('data/SiO2-Rodriguez.yml', WLs, "nm")

    Q_sca = []
    Q_ext = []
    if len(WLs) == 1:
        index_Si = [index_Si]
        index_substrate = [index_substrate]
    paramlist = []
    for cylinder_R in cylinder_Rs:
        for i in range(len(WLs)):
            index_NP = 3.94 #index_Si[i][1]
            index_subst = 1.45 #index_substrate[i][1]
            index_subst2 = 1.45 #index_Si[i][1]
            # if mpi_rank == 0:
            #     print("===> Params: WL ", WLs[i], ",\tR ", cylinder_R, ",\tn", index_NP,)
            paramlist.append([WLs[i], index_NP, index_subst, index_subst2, cylinder_R, polarization])
            #valTM = GetTopTSCS(WLs[i], index_NP, index_subst, cylinder_R, TM)
            #valTE, valTE_ext = GetTopTSCS(WLs[i], index_NP, index_subst, cylinder_R, polarization=TE)
    #print("My rank ",mpi_rank, "of", mpi_size)
    PrintProgress(1)
    result = []
    for i in range(len(paramlist)):
        if i % mpi_size != mpi_rank:
            continue
        progress = i*100.0/len(paramlist)
        if progress > 1: PrintProgress(progress)
        result.append(call_worker(paramlist[i]))
    result_np = np.array(result,dtype=np.float64)
    PrintProgress(100)
    sys.stdout = sys.__stdout__ # Restore output after muting it in simulation
    recvbuf = None
    if mpi_rank == 0:
        recvbuf = np.zeros([mpi_size,int(len(paramlist)/mpi_size)+1, len(result_np[0])],dtype=np.float64)
    sys.__stdout__.write("Gatering...\n")
    sys.__stdout__.flush()
    mpi_comm.Gather(result_np, recvbuf, root=0)
    # Sync all MPI processes here to avoid gathering from subsequent
    # call of main()
    mpi_comm.Barrier() 

    if mpi_rank == 0:
        result = recvbuf[-1]
        for i in range(len(recvbuf)-1):
            result = np.vstack((result, recvbuf[i]))
        # print("buf",recvbuf)
        
        result = result[~np.all(result == 0, axis=1)]
        print("Combined", len(result),"results of", len(paramlist))
        ind = np.lexsort((result[:,1],result[:,0]))    
        combined_plot(result[ind], polarization)
        
        # Q_sca.append(np.hstack((valTE, valTM)))
            # Q_ext.append(np.hstack((valTE_ext, valTM_ext)))
            # print("\nWL =", WLs[i],"(from", WLs[0],"to",WLs[-1],")  Q_sca = ", Q_sca[-1],"\n")
            # Q_sca1 = np.array(Q_sca)
            # print(Q_sca1)
    # plt.plot( WLs, Q_sca1)
    # plt.legend(["sca TE","sca TM"])
    # plt.xlabel(r'$\lambda, nm$')
    # plt.ylabel(r'$Q_{sca}$')
    # plt.title(' D = '+str(cylinder_R*2)+' nm')
    # plt.savefig(output_directory+"/Sca_cylinder_on_sapphire.png")
    # plt.clf()
    # plt.close()

    # Q_ext1 = np.array(Q_ext)
    # print(Q_ext1)
    # plt.plot( WLs, Q_ext1)
    # plt.legend(["ext TE","ext TM"])
    # plt.xlabel(r'$\lambda, nm$')
    # plt.ylabel(r'$Q_{ext}$')
    # plt.title(' D = '+str(cylinder_R*2)+' nm')
    # plt.savefig(output_directory+"/Ext_cylinder_on_sapphire.png")
    # plt.clf()
    # plt.close()

# PRIMES = [
#     112272535095293,
#     1099726899285419]

# def is_prime(n):
#     if n % 2 == 0:
#         return False
#     return True

# def main():

# This is just because I run the simulation on Windows HPC cluster,
# and do image postprocessing on linux...
if not sys.platform.startswith('linux'):
    main(TM)
    #main(TE)
else:
    if mpi_rank==0:
        #read_zarina()
        plot_file()
