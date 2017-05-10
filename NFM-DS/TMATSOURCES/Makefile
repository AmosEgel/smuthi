APPLICATION = TMATRIX

OBJS =  Parameters.o\
        MachParam.o\
	Allocation.o\
        AdditonTh.o\
        BesLeg.o\
        Check.o\
        SVWF.o\
	GeomLib.o\
        GeomTrans.o\
        IncCoeff.o\
        InputOutput.o\
        Integr.o\
        Interp.o\
	MatrixSolv.o\
        MatrixOp.o\
        MatrixQ.o\
	MatrixTrans.o\
	PostProces1.o\
        PostProces2.o\
        PostProces3.o\
        Proces1.o\
        Proces2.o\
        Proces3.o\
        Random.o\
        TAXSYM.o\
        TCOMP.o\
        TINHOM2SPH.o\
        TINHOM.o\
        TINHOMSPH.o\
        TINHOMSPHREC.o\
        TLAY.o\
        TMULT2SPH.o\
        TMULT.o\
        TMULTSPH.o\
        TMULTSPHREC.o\
        TNONAXSYM.o\
        TPARTSUB.o\
        TSPHERE.o\
        SCTAVRGSPH.o\
        SCT.o\
        EFMED.o\
        TMATRIX.o	

#Compiler option
FC = gfortran
#FC = ifort
#FC = /home/donau101/adsm/scia/bin/f95 -C -g -gline
#FC = f90

$(APPLICATION): $(OBJS) 
		$(FC) $(OBJS) -o $(APPLICATION)

Parameters.o:   Parameters.f90
		$(FC) Parameters.f90 -c
		
MachParam.o:    Parameters.o               MachParam.f90		
		$(FC) MachParam.f90 -c
		
Allocation.o:   Allocation.f90
		$(FC) Allocation.f90 -c		
		
AdditonTh.o:    Parameters.o  MachParam.o  AdditonTh.f90
	 	$(FC) AdditonTh.f90 -c 	
		
BesLeg.o:       Parameters.o  MachParam.o  BesLeg.f90
		$(FC) BesLeg.f90 -c
		
Check.o:        Parameters.o  MachParam.o  Check.f90
		$(FC) Check.f90 -c
		
SVWF.o:         Parameters.o  MachParam.o  SVWF.f90
		$(FC) SVWF.f90 -c
		
GeomLib.o:      Parameters.o  MachParam.o  GeomLib.f90
		$(FC) GeomLib.f90 -c
		
GeomTrans.o:    Parameters.o  MachParam.o  GeomTrans.f90
		$(FC) GeomTrans.f90 -c
		
IncCoeff.o:     Parameters.o  MachParam.o  IncCoeff.f90
		$(FC) IncCoeff.f90 -c
		
InputOutput.o:  Parameters.o  MachParam.o  InputOutput.f90
		$(FC) InputOutput.f90 -c
		
Integr.o:       Parameters.o  MachParam.o  Integr.f90
		$(FC) Integr.f90 -c
		
Interp.o:       Parameters.o  MachParam.o  Interp.f90
		$(FC) Interp.f90 -c
		
MatrixSolv.o:   Parameters.o  MachParam.o  MatrixSolv.f90
		$(FC) MatrixSolv.f90 -c		
		
MatrixOp.o:     Parameters.o  MachParam.o  MatrixOp.f90
		$(FC) MatrixOp.f90 -c
		
MatrixQ.o:      Parameters.o  MachParam.o  MatrixQ.f90
		$(FC) MatrixQ.f90 -c
		
MatrixTrans.o:  Parameters.o  MachParam.o  MatrixTrans.f90 
		$(FC) MatrixTrans.f90 -c
		
PostProces1.o:  Parameters.o  MachParam.o  PostProces1.f90
		$(FC) PostProces1.f90 -c
		
PostProces2.o:  Parameters.o  MachParam.o  PostProces2.f90
		$(FC) PostProces2.f90 -c
		
PostProces3.o:  Parameters.o  MachParam.o  PostProces3.f90
		$(FC) PostProces3.f90 -c
		
Proces1.o:      Parameters.o  MachParam.o  Proces1.f90
		$(FC) Proces1.f90 -c
		
Proces2.o:      Parameters.o  MachParam.o  Proces2.f90
		$(FC) Proces2.f90 -c
		
Proces3.o:      Parameters.o  MachParam.o  Proces3.f90
		$(FC) Proces3.f90 -c
		
Random.o:       Parameters.o  MachParam.o  Random.f90
		$(FC) Random.f90 -c

TAXSYM.o:       Parameters.o  MachParam.o  TAXSYM.f90
		$(FC) TAXSYM.f90 -c
		
TCOMP.o:        Parameters.o  MachParam.o  TCOMP.f90
		$(FC) TCOMP.f90 -c
		
TINHOM2SPH.o:   Parameters.o  MachParam.o  TINHOM2SPH.f90
		$(FC) TINHOM2SPH.f90 -c
		
TINHOM.o:       Parameters.o  MachParam.o  TINHOM.f90
		$(FC) TINHOM.f90 -c
		
TINHOMSPH.o:    Parameters.o  MachParam.o  TINHOMSPH.f90
		$(FC) TINHOMSPH.f90 -c
		
TINHOMSPHREC.o: Parameters.o  MachParam.o  TINHOMSPHREC.f90
		$(FC) TINHOMSPHREC.f90 -c
		
TLAY.o:         Parameters.o  MachParam.o  TLAY.f90
		$(FC) TLAY.f90 -c
		
TMULT2SPH.o:    Parameters.o  MachParam.o  TMULT2SPH.f90
		$(FC) TMULT2SPH.f90 -c
		
TMULT.o:        Parameters.o  MachParam.o  TMULT.f90
		$(FC) TMULT.f90 -c
		
TMULTSPH.o:     Parameters.o  MachParam.o  TMULTSPH.f90
		$(FC) TMULTSPH.f90 -c
		
TMULTSPHREC.o:  Parameters.o  MachParam.o  TMULTSPHREC.f90
		$(FC) TMULTSPHREC.f90 -c
		
TNONAXSYM.o:    Parameters.o  MachParam.o  TNONAXSYM.f90
		$(FC) TNONAXSYM.f90 -c
		
TPARTSUB.o:     Parameters.o  MachParam.o  TPARTSUB.f90
		$(FC) TPARTSUB.f90 -c
		
TSPHERE.o:      Parameters.o  MachParam.o  TSPHERE.f90
		$(FC) TSPHERE.f90 -c
		
SCTAVRGSPH.o:   Parameters.o  MachParam.o  SCTAVRGSPH.f90
		$(FC) SCTAVRGSPH.f90 -c
		
SCT.o:          Parameters.o  MachParam.o  SCT.f90
		$(FC) SCT.f90 -c
		
EFMED.o:        Parameters.o  MachParam.o  EFMED.f90
		$(FC) EFMED.f90 -c

TMATRIX.o:      Parameters.o  MachParam.o  TMATRIX.f90
		$(FC) TMATRIX.f90 -c


clean:		rm -f $(APPLICATION) $(OBJS) *.mod
