# Test CUDA
#cmake_minimum_required(VERSION 3.25)
#project(myapp)

#set(CMAKE_CXX_STANDARD 20)

#find_package(CUDAToolkit REQUIRED)
#add_executable(${PROJECT_NAME} test.cpp)

#target_link_libraries(${PROJECT_NAME} CUDA::cudart)
#
##Check usual algorithm
#
#cmake_minimum_required(VERSION 3.25)
#
#project(GOST CXX CUDA)
#
#set(CMAKE_CXX_STANDARD 20)
#
#find_package(CUDAToolkit REQUIRED)
#
#add_executable(${PROJECT_NAME} main.cpp 
#    ./structures/structures.cuh
#    ./magma/magma.cuh
#    ./magma/magma.cu
#    ./kuznechik/kuznechik.cuh
#    ./kuznechik/kuznechik.cu
#    getInfoOfDevice.cu
#    )
#
#target_link_libraries(${PROJECT_NAME} CUDA::cudart)

#OpenSSL provider
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

set(BUILD_SHARED_LIBS ON)

project(gost_provider CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(CUDAToolkit REQUIRED)
link_libraries(${PROJECT_NAME} CUDA::cudart) 

add_subdirectory(libprov)
include(libprov/cmake/provider.cmake)

setup_provider_openssl()

include_directories("${CUDAToolkit_INCLUDE_DIRS}")

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/gParams.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gParams.h
                   COMMAND perl ${CMAKE_CURRENT_SOURCE_DIR}/libprov/perl/gen_param_LL.pl
                           ${CMAKE_CURRENT_SOURCE_DIR}/gParams.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/gParams.h
                           ${CMAKE_CURRENT_SOURCE_DIR}/gostParams.dat
                   MAIN_DEPENDENCY gostParams.dat
                   DEPENDS libprov/perl/gen_param_LL.pl)

set(gost_source myProvider.cpp gParams.cpp ./magma/magma.cu ./magmaContext/magmaContext.cpp 
        ./kuznechik/kuznechik.cu ./kuznechikContext/kuznechikContext.cpp)

build_provider(gost_provider "${gost_source}" libprov)