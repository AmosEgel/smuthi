module parameters
  implicit none
  integer,parameter       :: O = kind(1.d0)
!
  real(O),parameter       :: Pi    = 3.1415926535897932384626433832795028841971_O
  complex(O),parameter    :: im    = (0._O,1._O),                                        &
                             one   = (1._O,0._O),                                        &
                             zero  = (0._O,0._O)
!
  integer,parameter       :: NsurfPD =     10,                                           &
                             NrankPD =    200,                                           &
                             NfacePD =  10000,                                           &
                             NparPD  =     10,                                           &
                             NphiMax =    361
!
  integer,parameter       :: iTmat        =  8,                                          &
                             iTmatInfo    =  9,                                          &
                             iDSCS        = 10,                                          &
                             iSCAT        = 11,                                          &
                             iSS          = 12,                                          & 			                               
                             iFEM         = 13
!
  integer,parameter       :: iamat = 14,                                                 &
                             ibmat = 15,                                                 &
                             icmat = 16,                                                 &
                             idmat = 17,                                                 &
                             iemat = 18,                                                 &
                             ifmat = 19
!
  integer,parameter       :: iOutput           = 20,                                     &
                             iInput            = 21,                                     &
                             iInputAXSYM       = 22,                                     &
                             iInputNONAXSYM    = 23,                                     &
                             iInputNONAXSYMFEM = 24,                                     &
                             iInputCOMP        = 25,                                     &
                             iInputLAY         = 26,                                     &
!
                             iInputINHOM       = 27,                                     &
                             iInputINHOM2SPH   = 28,                                     &
                             iInputINHOMSPH    = 29,                                     &
                             iInputINHOMSPHREC = 30,                                     &
!
                             iInputMULT        = 31,                                     &
                             iInputMULT2SPH    = 32,                                     &
                             iInputMULTSPH     = 33,                                     &
                             iInputMULTSPHREC  = 34,                                     &
!
                             iInputSPHERE      = 35,                                     &
                             iInputPARTSUB     = 36,                                     &
                             iInputANIS        = 37,                                     &
                             iInputEFMED       = 38,                                     &
!
                             iInputSCT         = 39,                                     &
                             iInputSCTAVRGSPH  = 40
!
  character(80),parameter ::                                                             &
! The following lines were removed from the original code: >>>>>>>>>>>>>>>>>>>>>>>>>>
!               FileOutput           = "../OUTPUTFILES/Output.dat",                      &
!               FileInput            = "../INPUTFILES/Input.dat",                        &
!               FileInputAXSYM       = "../INPUTFILES/InputAXSYM.dat",                   &
!                FileInputNONAXSYM    = "../INPUTFILES/InputNONAXSYM.dat",                &
!                FileInputNONAXSYMFEM = "../INPUTFILES/InputNONAXSYMFEM.dat",             &
!                FileInputCOMP        = "../INPUTFILES/InputCOMP.dat",                    &
!                FileInputLAY         = "../INPUTFILES/InputLAY.dat",                     &
!!
!                FileInputINHOM       = "../INPUTFILES/InputINHOM.dat",                   &
!                FileInputINHOM2SPH   = "../INPUTFILES/InputINHOM2SPH.dat",               & 
!                FileInputINHOMSPH    = "../INPUTFILES/InputINHOMSPH.dat" ,               &
!                FileInputINHOMSPHREC = "../INPUTFILES/InputINHOMSPHREC.dat",             &
!!               			
!                FileInputMULT        = "../INPUTFILES/InputMULT.dat",                    &
!                FileInputMULT2SPH    = "../INPUTFILES/InputMULT2SPH.dat" ,               &
!                FileInputMULTSPH     = "../INPUTFILES/InputMULTSPH.dat",                 &
!                FileInputMULTSPHREC  = "../INPUTFILES/InputMULTSPHREC.dat",              &
!!               									
!                FileInputSPHERE      = "../INPUTFILES/InputSPHERE.dat",                  &
!                FileInputPARTSUB     = "../INPUTFILES/InputPARTSUB.dat",                 &
!                FileInputANIS        = "../INPUTFILES/InputANIS.dat",                    &
!                FileInputEFMED       = "../INPUTFILES/InputEFMED.dat",                   &
!!
!                FileInputSCT         = "../INPUTFILES/InputSCT.dat",                     &
!                FileInputSCTAVRGSPH  = "../INPUTFILES/InputSCTAVRGSPH.dat"
!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
! The following lines were added to the original code: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                FileOutput           = "smuthi/nfmds/data/Output.dat",             &
                FileInput            = "smuthi/nfmds/data/Input.dat",                    &
                FileInputAXSYM       = "smuthi/nfmds/data/InputAXSYM.dat",               &
                FileInputNONAXSYM    = "smuthi/nfmds/data/InputNONAXSYM.dat",                &
                FileInputNONAXSYMFEM = "smuthi/nfmds/data/InputNONAXSYMFEM.dat",             &
                FileInputCOMP        = "smuthi/nfmds/data/InputCOMP.dat",                    &
                FileInputLAY         = "smuthi/nfmds/data/InputLAY.dat",                     &
!
                FileInputINHOM       = "smuthi/nfmds/data/InputINHOM.dat",                   &
                FileInputINHOM2SPH   = "smuthi/nfmds/data/InputINHOM2SPH.dat",               & 
                FileInputINHOMSPH    = "smuthi/nfmds/data/InputINHOMSPH.dat" ,               &
                FileInputINHOMSPHREC = "smuthi/nfmds/data/InputINHOMSPHREC.dat",             &
!               			
                FileInputMULT        = "smuthi/nfmds/data/InputMULT.dat",                    &
                FileInputMULT2SPH    = "smuthi/nfmds/data/InputMULT2SPH.dat" ,               &
                FileInputMULTSPH     = "smuthi/nfmds/data/InputMULTSPH.dat",                 &
                FileInputMULTSPHREC  = "smuthi/nfmds/data/InputMULTSPHREC.dat",              &
!               									
                FileInputSPHERE      = "smuthi/nfmds/data/InputSPHERE.dat",                  &
                FileInputPARTSUB     = "smuthi/nfmds/data/InputPARTSUB.dat",                 &
                FileInputANIS        = "smuthi/nfmds/data/InputANIS.dat",                    &
                FileInputEFMED       = "smuthi/nfmds/data/InputEFMED.dat",                   &
!
                FileInputSCT         = "smuthi/nfmds/data/InputSCT.dat",                     &
                FileInputSCTAVRGSPH  = "smuthi/nfmds/data/InputSCTAVRGSPH.dat"
!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
!
! The following lines were removed from the original code: >>>>>>>>>>>>>>>>>>>>>>>>>>
!  character(15),parameter :: PathOUTPUT = "../OUTPUTFILES/",                            &
!                             PathTEMP   = "../TEMPFILES/",                              &
!                             PathGEOM   = "../GEOMFILES/"                       
!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
! The following lines were added to the original code: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  character(15),parameter :: PathOUTPUT = "smuthi/nfmds/data/",                          &
                             PathTEMP   = "smuthi/nfmds/temp/",                          &
                             PathGEOM   = "smuthi/nfmds/data/"
!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<							 
end module parameters
! **************************************************************************************
module derived_parameters
  use parameters
  implicit none
!  
  integer,save :: NBaseDig
  integer,save :: NIterBes
  integer,save :: NIterPol
  real(O),save :: LargestPosNumber
  real(O),save :: SmallestPosNumber
  real(O),save :: MachEps           
  real(O),save :: ZeroCoord
  real(O),save :: TolJ0Val 
  real(O),save :: TolRootPol
  real(O),save :: InitBesVal
  real(O),save :: FactNBes
  real(O),save :: LargestBesVal
  real(O),save :: MaxArgBes
  real(O),save :: UpperBoundSeq 
  real(O),save :: LowerBoundSeq   
  real(O),save :: ZeroSinXX  
  real(O),save :: ZeroLUVal 
  real(O),save :: LargestSplineVal  
end module derived_parameters
                      
