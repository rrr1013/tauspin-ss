atlas_subdir(TauSpinNtuple)

atlas_add_library(
    TauSpinNtupleLib
    TauSpinNtuple/*.h
    Root/*.cxx
    PUBLIC_HEADERS TauSpinNtuple
    LINK_LIBRARIES
        AnaAlgorithmLib
        SelectionHelpersLib
        SystematicsHandlesLib
        xAODTau
)

atlas_add_dictionary(
    TauSpinNtupleDict
    TauSpinNtuple/TauSpinNtupleDict.h
    TauSpinNtuple/selection.xml
    LINK_LIBRARIES
        TauSpinNtupleLib
)

atlas_install_data(data/*.yaml)