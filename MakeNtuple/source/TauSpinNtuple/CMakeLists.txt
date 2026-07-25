atlas_subdir(TauSpinNtuple)

atlas_add_library(
    TauSpinNtupleLib
    TauSpinNtuple/*.h
    Root/*.cxx
    PUBLIC_HEADERS TauSpinNtuple
    LINK_LIBRARIES
        AnaAlgorithmLib
)