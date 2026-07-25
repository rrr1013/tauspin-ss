cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(TauSpinSS VERSION 1.0.0)

find_package(AnalysisBase REQUIRED)

atlas_ctest_setup()

atlas_project(
    TauSpinSS
    USE AnalysisBase ${AnalysisBase_VERSION}
)

lcg_generate_env(
    SH_FILE ${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/env_setup.sh
)

install(
    FILES ${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/env_setup.sh
    DESTINATION .
)

atlas_cpack_setup()